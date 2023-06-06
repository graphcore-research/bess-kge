# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, cast

import einops
import numpy as np
from numpy.typing import NDArray

from besskge.sharding import Sharding


class ShardedNegativeSampler(ABC):
    """
    Base class for sharded negative sampler.
    """

    #: Sample negatives per triple partition, instead of per triple
    flat_negative_format: bool
    #: Sample negatives only from processing device
    local_sampling: bool
    #: Which entity to corrupt; "h", "t", "ht"
    corruption_scheme: str

    @abstractmethod
    def __call__(
        self,
        sample_idx: NDArray[np.int64],
    ) -> Dict[str, Union[NDArray[np.int32], NDArray[np.bool_]]]:
        """
        Sample negatives for batch.

        :param sample_idx: shape: (bps, n_shard, [n_shard,] triple_per_partition)
            Per-partition indices of triples in batch (for all bps batches in a step).

        :return: "negative_entities" shape: (bps, n_shard, n_shard, B, n_negative)
                B = 1 if :attr:`flat_negative_format`, :attr:`corruption_scheme`=="h","t"
                B = 2 if :attr:`flat_negative_format`, :attr:`corruption_scheme`=="ht"
                else B = shard_bs
            Negative samples for triples in batch.
            The elements in `(*, shard_source, shard_dest, *, *)`
            are the negative samples to collect from `shard_source`
            and use for the batch on `shard_dest` (if :attr:`local_sampling` = False,
            otherwise on `shard_source`).
            + other relevant data.
        """
        raise NotImplementedError


class RandomShardedNegativeSampler(ShardedNegativeSampler):
    """
    Sample random negative entities for each triple in batch.
    """

    def __init__(
        self,
        n_negative: int,
        sharding: Sharding,
        seed: int,
        corruption_scheme: str,
        local_sampling: bool,
        flat_negative_format: bool = False,
    ) -> None:
        """
        Initialize random negative sampler.

        :param n_negative:
            Number of negative samples per shard-pair
            (if :attr:`flat_negative_format`) or per triple.
        :param sharding:
            Sharding of entities.
        :param seed:
            Seed of RNG.
        :param corruption_scheme:
            "h": corrupt head entities;
            "t": corrupt tail entities;
            "ht": corrupt head entities for the first half of each triple partition,
            tail entities for the second half.
        :param local_sampling:
            Sample negative entities only from the shard where the triple is processed.
        :param flat_negative_format:
            Sample :attr:`n_negative` negative entities for each shard-pair,
            instead of each triple. If True, requires use of negative sample
            sharing. Defaults to False.
        """
        self.n_negative = n_negative
        self.sharding = sharding
        self.shard_counts = sharding.shard_counts
        self.corruption_scheme = corruption_scheme
        self.local_sampling = local_sampling
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.flat_negative_format = flat_negative_format

    # docstr-coverage: inherited
    def __call__(
        self,
        sample_idx: NDArray[np.int64],
    ) -> Dict[str, Union[NDArray[np.int32], NDArray[np.bool_]]]:
        batches_per_step, n_shard = sample_idx.shape[:2]
        positive_per_partition = sample_idx.shape[-1]
        shard_bs = (
            positive_per_partition
            if len(sample_idx.shape) == 3
            else n_shard * positive_per_partition
        )
        if self.flat_negative_format:
            B = 2 if self.corruption_scheme == "ht" else 1
        else:
            B = shard_bs
        negative_entities = (
            self.rng.integers(
                1 << 31,
                size=(
                    batches_per_step,
                    n_shard,
                    n_shard,
                    B,
                    self.n_negative,
                ),
            ).astype(np.int32)
            % self.shard_counts[None, :, None, None, None]
        )
        return dict(negative_entities=negative_entities)


class TypeBasedShardedNegativeSampler(RandomShardedNegativeSampler):
    """
    Corrupt entities with entities of the same type.
    """

    def __init__(
        self,
        triple_types: NDArray[np.int32],
        n_negative: int,
        sharding: Sharding,
        corruption_scheme: str,
        local_sampling: bool,
        seed: int,
    ) -> None:
        """
        Initialize type-based negative sampler.

        :param triple_types: shape: (n_triple, 2)
            Type IDs of head and tail entities for all triples.
        :param n_negative:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param sharding:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param corruption_scheme:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param local_sampling:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param seed:
            see :meth:`RandomShardedNegativeSampler.__init__`
        """
        super(TypeBasedShardedNegativeSampler, self).__init__(
            n_negative,
            sharding,
            seed,
            corruption_scheme,
            local_sampling,
            flat_negative_format=False,
        )
        self.triple_types = triple_types
        if sharding.entity_type_counts is None or sharding.entity_type_offsets is None:
            raise ValueError("The provided entity sharding does not have entity types")
        self.type_offsets = sharding.entity_type_offsets
        self.type_counts = sharding.entity_type_counts

    # docstr-coverage: inherited
    def __call__(
        self,
        sample_idx: NDArray[np.int64],
    ) -> Dict[str, Union[NDArray[np.int32], NDArray[np.bool_]]]:
        n_shard = sample_idx.shape[1]
        positive_per_partition = sample_idx.shape[-1]
        head_type, tail_type = einops.rearrange(
            self.triple_types[sample_idx],
            "... ht -> ht ...",
        )

        if self.corruption_scheme == "h":
            relevant_type = head_type
        elif self.corruption_scheme == "t":
            relevant_type = tail_type
        elif self.corruption_scheme == "ht":
            cut_point = positive_per_partition // 2
            relevant_type = np.concatenate(
                [head_type[..., :cut_point], tail_type[..., cut_point:]], axis=-1
            )
        else:
            raise ValueError(
                f"Corruption scheme {self.corruption_scheme}"
                " not supported by {self.__class__}"
            )

        if self.local_sampling:
            repeat_pattern = "step shard ... triple -> step shard r (... triple)"
        else:
            repeat_pattern = "step shard ... triple -> step r shard (... triple)"
        relevant_type = einops.repeat(
            relevant_type,
            repeat_pattern,
            r=n_shard,
        )

        rvs = super(TypeBasedShardedNegativeSampler, self).__call__(
            sample_idx,
        )["negative_entities"]

        negative_entities = (
            rvs
            % self.type_counts[
                np.arange(n_shard)[None, :, None, None], relevant_type, np.newaxis
            ]
            + self.type_offsets[
                np.arange(n_shard)[None, :, None, None], relevant_type, np.newaxis
            ]
        )

        return dict(negative_entities=negative_entities)


class TripleBasedShardedNegativeSampler(ShardedNegativeSampler):
    """
    Return (possibly triple-specific) predetermined negative entities.
    """

    def __init__(
        self,
        negative_heads: Optional[NDArray[np.int32]],
        negative_tails: Optional[NDArray[np.int32]],
        sharding: Sharding,
        corruption_scheme: str,
        seed: int,
        mask_on_gather: bool = False,
        return_sort_idx: bool = False,
    ):
        """
        Initialize triple-based negative sampler.

        :param negative_heads: shape: (N, n_negative)
            Global entity IDs of negative heads, specific
            for each triple (N=n_triple) or for all of them (N=1).
        :param negative_tails: shape: (N, n_negative)
            Global entity IDs of negative tails, specific
            for each triple (N=n_triple) or for all of them (N=1).
        :param sharding:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param corruption_scheme:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param seed:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param mask_on_gather:
            Shape the negative mask to be applied on the device where
            negative entities are gathered, instead of the one where they
            are scored. Set to `True` only when using
            :class:`besskge.bess.TopKQueryBessKGE`. Defaults to False.
        :param return_sort_idx:
            Return for each triple in batch the sorting indices
            to recover same ordering of negatives as in :attr:`negative_heads`,
            :attr:`negative_tails`. Defaults to False.
        """
        self.N: int
        self.n_negative: int
        if negative_heads is not None and negative_tails is not None:
            assert (
                negative_heads.shape == negative_tails.shape
            ), "negative_heads and negative_tails need to have same size"
            negative_heads = negative_heads.reshape(-1, negative_heads.shape[-1])
            negative_tails = negative_tails.reshape(-1, negative_tails.shape[-1])
            self.N, self.n_negative = negative_heads.shape
        elif negative_tails is not None:
            assert corruption_scheme == "t", (
                f"Corruption scheme '{corruption_scheme}' requires"
                " providing negative_heads"
            )
            negative_tails = negative_tails.reshape(-1, negative_tails.shape[-1])
            self.N, self.n_negative = negative_tails.shape
        elif negative_heads is not None:
            assert corruption_scheme == "h", (
                f"Corruption scheme '{corruption_scheme}' requires"
                " providing negative_tails"
            )
            negative_heads = negative_heads.reshape(-1, negative_heads.shape[-1])
            self.N, self.n_negative = negative_heads.shape
        else:
            raise ValueError(
                "At least one between negative_heads and negative_tails"
                " needs to be provided"
            )

        self.sharding = sharding
        self.shard_counts = sharding.shard_counts
        self.corruption_scheme = corruption_scheme
        self.local_sampling = False
        self.flat_negative_format = self.N == 1
        self.return_sort_idx = return_sort_idx
        self.rng = np.random.default_rng(seed=seed)

        if self.corruption_scheme in ["h", "t"]:
            negatives = cast(
                NDArray[np.int32],
                negative_heads if self.corruption_scheme == "h" else negative_tails,
            )  # mypy check
            (
                shard_neg_counts,
                self.sort_neg_idx,
            ) = self.shard_negatives(negatives)
            self.padded_shard_length = shard_neg_counts.max()
            self.padded_negatives, self.mask = self.pad_negatives(
                sharding.entity_to_idx[
                    np.take_along_axis(negatives, self.sort_neg_idx, axis=-1)
                ],
                shard_neg_counts,
                self.padded_shard_length,
            )
        elif self.corruption_scheme == "ht":
            negative_heads = cast(NDArray[np.int32], negative_heads)  # mypy check
            negative_tails = cast(NDArray[np.int32], negative_tails)  # mypy check
            (
                shard_neg_h_counts,
                self.sort_neg_h_idx,
            ) = self.shard_negatives(negative_heads)
            (
                shard_neg_t_counts,
                self.sort_neg_t_idx,
            ) = self.shard_negatives(negative_tails)
            self.padded_shard_length = np.max(
                [shard_neg_h_counts.max(), shard_neg_t_counts.max()]
            )
            self.padded_negatives_h, self.mask_h = self.pad_negatives(
                sharding.entity_to_idx[
                    np.take_along_axis(negative_heads, self.sort_neg_h_idx, axis=-1)
                ],
                shard_neg_h_counts,
                self.padded_shard_length,
            )
            self.padded_negatives_t, self.mask_t = self.pad_negatives(
                sharding.entity_to_idx[
                    np.take_along_axis(negative_tails, self.sort_neg_t_idx, axis=-1)
                ],
                shard_neg_t_counts,
                self.padded_shard_length,
            )
        else:
            raise ValueError(
                f"Corruption scheme {self.corruption_scheme}"
                " not supported by {self.__class__}"
            )

        # Negative entities are consumed on gathering device (shard_neg)
        self.ent_rearrange_pattern = (
            "step shard ... triple shard_neg idx_neg ->"
            "step shard_neg shard (... triple) idx_neg"
        )
        self.ent_repeat_pattern = (
            "pad shard_neg idx_neg -> step shard_neg shard pad idx_neg"
        )
        self.mask_on_gather = mask_on_gather
        if self.mask_on_gather:
            # Negative masks are consumed on gathering device (shard_neg)
            self.mask_rearrange_pattern = self.ent_rearrange_pattern
            self.mask_repeat_pattern = self.ent_repeat_pattern
        else:
            # Negative masks are consumed on processing device (shard)
            self.mask_rearrange_pattern = (
                "step shard ... triple shard_neg idx_neg ->"
                "step shard (... triple) shard_neg idx_neg"
            )
            self.mask_repeat_pattern = (
                "pad shard_neg idx_neg -> step shard pad shard_neg idx_neg"
            )

    # docstr-coverage: inherited
    def __call__(
        self,
        sample_idx: NDArray[np.int64],
    ) -> Dict[str, Union[NDArray[np.int32], NDArray[np.bool_]]]:
        if self.corruption_scheme in ["h", "t"]:
            if self.flat_negative_format:
                sample_idx_orig_shape = sample_idx.shape
                sample_idx = np.full(fill_value=0, shape=(*sample_idx.shape[:2], 1))

            negative_entities = einops.rearrange(
                self.padded_negatives[sample_idx],
                self.ent_rearrange_pattern,
            )
            negative_mask = einops.rearrange(
                self.mask[sample_idx],
                self.mask_rearrange_pattern,
            )
            if self.return_sort_idx:
                if self.flat_negative_format:
                    negative_sort_idx = self.sort_neg_idx[
                        np.full(fill_value=0, shape=sample_idx_orig_shape)
                    ]
                else:
                    negative_sort_idx = self.sort_neg_idx[sample_idx]
        elif self.corruption_scheme == "ht":
            cutpoint = sample_idx.shape[-1] // 2
            if self.flat_negative_format:
                bps, n_shard = sample_idx.shape[:2]
                negative_entities = einops.repeat(
                    np.concatenate(
                        [self.padded_negatives_h, self.padded_negatives_t], axis=0
                    ),
                    self.ent_repeat_pattern,
                    step=bps,
                    shard=n_shard,
                )
                negative_mask = einops.repeat(
                    np.concatenate([self.mask_h, self.mask_t], axis=0),
                    self.mask_repeat_pattern,
                    step=bps,
                    shard=n_shard,
                )
                if self.return_sort_idx:
                    sample_h_idx = np.full(
                        fill_value=0, shape=(*sample_idx.shape[:-1], cutpoint)
                    )
                    sample_t_idx = np.full(
                        fill_value=0,
                        shape=(*sample_idx.shape[:-1], sample_idx.shape[-1] - cutpoint),
                    )
            else:
                sample_h_idx = sample_idx[..., :cutpoint]
                sample_t_idx = sample_idx[..., cutpoint:]
                negative_entities = einops.rearrange(
                    np.concatenate(
                        [
                            self.padded_negatives_h[sample_h_idx],
                            self.padded_negatives_t[sample_t_idx],
                        ],
                        axis=-3,
                    ),
                    self.ent_rearrange_pattern,
                )
                negative_mask = einops.rearrange(
                    np.concatenate(
                        [
                            self.mask_h[sample_h_idx],
                            self.mask_t[sample_t_idx],
                        ],
                        axis=-3,
                    ),
                    self.mask_rearrange_pattern,
                )
            if self.return_sort_idx:
                negative_sort_idx = np.concatenate(
                    [
                        self.sort_neg_h_idx[sample_h_idx],
                        self.sort_neg_t_idx[sample_t_idx],
                    ],
                    axis=-2,
                )
        out_dict = dict(
            negative_entities=negative_entities,
            negative_mask=negative_mask,
        )
        if self.return_sort_idx:
            negative_sort_idx = einops.rearrange(
                negative_sort_idx,
                "step shard ... triple idx_neg -> step shard (... triple) idx_neg",
            )
            out_dict.update(negative_sort_idx=negative_sort_idx)

        return cast(Dict[str, Union[NDArray[np.int32], NDArray[np.bool_]]], out_dict)

    def shard_negatives(
        self,
        negatives: NDArray[np.int32],
    ) -> Tuple[NDArray[np.int64], NDArray[np.int32]]:
        """
        Split negative entities into correpsonding shards.

        :param negatives: shape: (N, n_negatives)
            Negative entities to shard (N = 1, n_triple).
        :return shard_neg_counts: shape: (N, n_shard)
            Number of negative entities per shard.
        :return sort_neg_idx: shape: (N, n_negatives)
            Sorting index to cluster negatives in shard order.
        """
        n_shard = self.sharding.n_shard
        shard_idx = self.sharding.entity_to_shard[negatives]
        shard_neg_counts = np.bincount(
            (shard_idx + n_shard * np.arange(self.N)[:, None]).flatten(),
            minlength=n_shard * self.N,
        ).reshape(self.N, n_shard)
        sort_neg_idx = np.argsort(shard_idx, axis=-1)

        return shard_neg_counts, sort_neg_idx.astype(np.int32)

    def pad_negatives(
        self,
        negatives: NDArray[np.int32],
        shard_counts: NDArray[np.int64],
        padded_shard_length: int,
    ) -> Tuple[NDArray[np.int32], NDArray[np.bool_]]:
        """
        Divide negatives based on shard and pad lists to same length.

        :param negatives: shape: (N, n_negative)
            Negative entities, each row already sorted in shard order
            (N = 1, n_triple).
        :param shard_counts: shape: (N, n_shard)
            Number of negatives per shard.
        :param padded_shard_length:
            The size to which each shard list is to be padded.

        :return padded_negatives: shape: (N, n_shard, padded_shard_length)
            The padded shard lists of negatives.
        :return mask: shape: (N, n_negative)
            Indices of true negatives in :code:`padded_negatives.view(N,-1)`.
        """
        mask_bool = (
            np.arange(padded_shard_length)[None, None, :] < shard_counts[..., None]
        )  # shape(N, n_shard, padded_shard_length)

        shard_offsets = np.c_[[0] * self.N, np.cumsum(shard_counts, axis=-1)[:, :-1]]

        shard_idx = (
            np.arange(padded_shard_length)[None, None, :] % shard_counts[..., None]
        )
        global_idx = np.minimum(
            shard_idx + shard_offsets[..., None], self.n_negative - 1
        )

        padded_negatives = negatives[np.arange(self.N)[:, None, None], global_idx]

        return padded_negatives, mask_bool


class PlaceholderNegativeSampler(ShardedNegativeSampler):
    """
    A placeholder sharded negative smapler, returns no negatives when called.
    Used with :class:`besskge.bess.TopKQueryBessKGE` to score queries against
    all entities in the KG.
    """

    def __init__(
        self,
        corruption_scheme: str,
        seed: int = 0,
    ) -> None:
        """
        Initialize placeholder negative sampler.

        :param corruption_scheme:
            see :class:`ShardedNegativeSampler`
        :param seed:
            No effect.
        """
        self.corruption_scheme = corruption_scheme
        self.local_sampling = False
        self.flat_negative_format = True
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    # docstr-coverage: inherited
    def __call__(
        self,
        sample_idx: NDArray[np.int64],
    ) -> Dict[str, Union[NDArray[np.int32], NDArray[np.bool_]]]:
        return dict()
