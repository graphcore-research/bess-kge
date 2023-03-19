# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import einops
import numpy as np

from besskge.dataset import Sharding


class ShardedNegativeSampler(ABC):
    """
    Base class for sharded negative sampler.
    """

    @abstractmethod
    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Sample negatives for batch.

        :param sample_idx: shape: (bps, n_shard, n_shard, triple_per_shardpair)
            Per-shardpair indices of triples in batch (for all bps steps).

        :return: shape: (bps, n_shard, n_shard, B, n_negative)
                with B = 1 if :attr:`flat_negative_format`
                else n_shard * triple_per_shardpair
            Negative samples for triples in batch.
            The elements in `(*, shard_source, *, *, *)` are the
            negative samples to collect from `shard_source`.
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
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize random negative sampler.

        :param n_negative:
            Number of negative samples per shardpair
            (if :attr:`flat_negative_format`) or per triple.
        :param sharding:
            Sharding of entities.
        :param seed:
            Seed of RNG.
        :param corruption_scheme:
            "h": corrupt head entities;
            "t": corrupt tail entities;
            "ht": corrupt head entities for the first half of each shardpair,
            tail entities for the second half.
        :param local_sampling:
            Sample negative entities from the local shard only.
        :param flat_negative_format:
            Sample :attr:`n_negative` negative entities for each shardpair,
            instead of each triple. If True, requires use of negative sample
            sharing. Defaults to False.
        """
        self.n_negative = n_negative
        self.sharding = sharding
        self.shard_counts = sharding.shard_counts
        self.corruption_scheme = corruption_scheme
        self.local_sampling = local_sampling
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.flat_negative_format = flat_negative_format
        # Triple-specific properties indexed by global triple ID
        self.triple_properties = []

    # docstr-coverage: inherited
    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        batches_per_step, n_shard, _, positive_per_shardpair = sample_idx.shape
        negative_entities = (
            self.rng.randint(
                1 << 63,
                size=(
                    batches_per_step,
                    n_shard,
                    n_shard,
                    1
                    if self.flat_negative_format
                    else n_shard * positive_per_shardpair,
                    self.n_negative,
                ),
            )
            % self.shard_counts[None, :, None, None, None]
        )
        return dict(negative_entities=negative_entities)


class TypeBasedShardedNegativeSampler(RandomShardedNegativeSampler):
    """Sample negative entities of the same type of the one to corrupt."""

    def __init__(
        self,
        triple_types: np.ndarray,
        n_negative: int,
        sharding: Sharding,
        corruption_scheme: str,
        local_sampling: bool,
        seed: int,
        *args,
        **kwargs,
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
            *args,
            **kwargs,
        )
        self.triple_types = triple_types
        self.type_offsets = sharding.entity_type_offsets
        self.type_counts = sharding.entity_type_counts
        self.triple_properties = ["triple_types"]

    # docstr-coverage: inherited
    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        _, n_shard, _, positive_per_shardpair = sample_idx.shape
        head_type, tail_type = einops.rearrange(
            self.triple_types[sample_idx],
            "step shard_h shard_t triple ht -> ht step shard_h shard_t triple",
        )

        if self.corruption_scheme == "h":
            relevant_type = head_type
        elif self.corruption_scheme == "t":
            relevant_type = tail_type
        elif self.corruption_scheme == "ht":
            cut_point = positive_per_shardpair // 2
            relevant_type = np.concatenate(
                [head_type[..., :cut_point], tail_type[..., cut_point:]], axis=-1
            )
        else:
            raise ValueError(
                f"Corruption scheme {self.corruption_scheme} not supported by {self.__class__}"
            )

        if self.local_sampling:
            repeat_pattern = (
                "step shard_h shard_t triple -> step shard_h repeat (shard_t triple)"
            )
        else:
            repeat_pattern = (
                "step shard_h shard_t triple -> step repeat shard_h (shard_t triple)"
            )
        relevant_type = einops.repeat(
            relevant_type,
            repeat_pattern,
            repeat=n_shard,
        )

        rvs = super(TypeBasedShardedNegativeSampler, self).get_negative_batch(
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
    Return triple-specific negative entities.
    """

    def __init__(
        self,
        negative_heads: np.ndarray,
        negative_tails: np.ndarray,
        # max_negative_per_step: int,
        sharding: Sharding,
        corruption_scheme: str,
        seed: int,
        get_sort_idx: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize triple-based negative sampler.

        :param negative_heads: shape: (n_triple, n_negatives)
            Global entity IDs of negative heads for each triple.
        :param negative_tails: shape: (n_triple, n_negatives)
            Global entity IDs of negative tails for each triple.
        :param sharding:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param corruption_scheme:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param seed:
            see :meth:`RandomShardedNegativeSampler.__init__`
        :param get_sort_idx:
            Provide for each triple in batch the sorting indices
            to recover same ordering of negatives as in :attr:`negative_heads`,
            :attr:`negative_tails`. Defaults to False.
        """
        if (
            negative_heads is not None
            and negative_tails is not None
            and negative_heads.shape != negative_tails.shape
        ):
            raise ValueError("negative heads and tails need to have the same dimension")
        self.n_triple, self.nss = negative_heads.shape
        # self.max_negative_per_step = max_negative_per_step
        self.sharding = sharding
        self.shard_counts = sharding.shard_counts
        self.corruption_scheme = corruption_scheme
        self.local_sampling = False
        self.flat_negative_format = False
        self.get_sort_idx = get_sort_idx
        self.rng = np.random.RandomState(seed=seed)

        if self.corruption_scheme in ["h", "t"]:
            negatives = (
                negative_heads if self.corruption_scheme == "h" else negative_tails
            )
            (
                shard_neg_counts,
                shard_neg_offsets,
                self.sort_neg_idx,
            ) = self.shard_negatives(negatives)
            self.padded_shard_length = shard_neg_counts.max()
            self.padded_negatives, self.mask = self.pad_negatives(
                sharding.entity_to_idx[
                    np.take_along_axis(negatives, self.sort_neg_idx, axis=-1)
                ],
                shard_neg_counts,
                shard_neg_offsets,
                self.padded_shard_length,
            )
            self.triple_properties = ["padded_negatives", "mask", "sort_neg_idx"]
        elif self.corruption_scheme == "ht":
            (
                shard_neg_h_counts,
                shard_neg_h_offsets,
                self.sort_neg_h_idx,
            ) = self.shard_negatives(negative_heads)
            (
                shard_neg_t_counts,
                shard_neg_t_offsets,
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
                shard_neg_h_offsets,
                self.padded_shard_length,
            )
            self.padded_negatives_t, self.mask_t = self.pad_negatives(
                sharding.entity_to_idx[
                    np.take_along_axis(negative_tails, self.sort_neg_t_idx, axis=-1)
                ],
                shard_neg_t_counts,
                shard_neg_t_offsets,
                self.padded_shard_length,
            )
            self.triple_properties = [
                "padded_negatives_h",
                "mask_h",
                "padded_negatives_t",
                "mask_t",
                "sort_neg_h_idx",
                "sort_neg_t_idx",
            ]
        else:
            raise ValueError(
                f"Corruption scheme {self.corruption_scheme} not supported by {self.__class__}"
            )

    # docstr-coverage: inherited
    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        if self.corruption_scheme in ["h", "t"]:
            sample_idx = einops.rearrange(
                sample_idx,
                "step shard_h shard_t triple -> step shard_h (shard_t triple)",
            )
            negative_entities = einops.rearrange(
                self.padded_negatives[sample_idx],
                "step shard_h triple shard_neg idx_neg-> step shard_neg shard_h triple idx_neg",
            )
            negative_mask = einops.rearrange(
                self.mask[sample_idx],
                "step shard_h triple shard_neg idx_neg-> step shard_h triple (shard_neg idx_neg)",
            )
            if self.get_sort_idx:
                negative_sort_idx = self.sort_neg_idx[sample_idx]
        elif self.corruption_scheme == "ht":
            cutpoint = sample_idx.shape[-1] // 2
            negative_entities = einops.rearrange(
                np.concatenate(
                    [
                        self.padded_negatives_h[sample_idx[:, :, :, :cutpoint]],
                        self.padded_negatives_t[sample_idx[:, :, :, cutpoint:]],
                    ],
                    axis=3,
                ),
                "step shard_h shard_t triple shard_neg idx_neg -> step shard_neg shard_h (shard_t triple) idx_neg",
            )
            negative_mask = einops.rearrange(
                np.concatenate(
                    [
                        self.mask_h[sample_idx[:, :, :, :cutpoint]],
                        self.mask_t[sample_idx[:, :, :, cutpoint:]],
                    ],
                    axis=3,
                ),
                "step shard_h shard_t triple shard_neg idx_neg -> step shard_h (shard_t triple) (shard_neg idx_neg)",
            )
            if self.get_sort_idx:
                negative_sort_idx = einops.rearrange(
                    np.concatenate(
                        [
                            self.sort_neg_h_idx[sample_idx[:, :, :, :cutpoint]],
                            self.sort_neg_t_idx[sample_idx[:, :, :, cutpoint:]],
                        ],
                        axis=3,
                    ),
                    "step shard_h shard_t triple idx_neg -> step shard_h (shard_t triple) idx_neg",
                )

        out_dict = dict(
            negative_entities=negative_entities,
            negative_mask=negative_mask,
        )
        if self.get_sort_idx:
            out_dict.update(negative_sort_idx=negative_sort_idx)

        return out_dict

    def shard_negatives(
        self,
        negatives: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """
        Split per-triple negative entities into correpsonding shards.

        :param negatives: shape: (n_triple, n_negatives)
            Negative entities to shard.
        :return shard_neg_counts: shape: (n_triple, n_shard)
            Number of negatives per shard for each triple.
        :return shard_neg_offsets: shape: (n_triple, n_shard)
            Offsets of shards after clustering negatives per shard.
        :return sort_neg_idx: shape: (n_triple, n_negatives)
            Per-triple sorting index to cluster negatives in shard order.
        """
        n_shard = self.sharding.n_shard
        shard_idx = self.sharding.entity_to_shard[negatives]
        shard_neg_counts = np.bincount(
            (shard_idx + n_shard * np.arange(self.n_triple)[:, None]).flatten(),
            minlength=n_shard * self.n_triple,
        ).reshape(self.n_triple, n_shard)
        shard_neg_offsets = np.c_[
            [0] * self.n_triple, np.cumsum(shard_neg_counts, axis=-1)[:, :-1]
        ]
        sort_neg_idx = np.argsort(shard_idx, axis=-1).astype(np.int32)

        return shard_neg_counts, shard_neg_offsets, sort_neg_idx

    def pad_negatives(
        self,
        negatives: np.ndarray,
        shard_counts: np.ndarray,
        shard_offsets: np.ndarray,
        padded_shard_length: np.ndarray,
    ) -> Tuple[np.ndarray]:
        """
        For each triple, divide negatives based on shard and pad lists to same length.

        :param negatives: shape: (n_triple, n_negatives)
            Per-triple negatives already sorted in shard order.
        :param shard_counts: shape: (n_triple, n_shard)
            Number of negatives per shard for each triple.
        :param shard_offsets: shape: (n_triple, n_shard)
            Offsets of shards for each triple.
        :param padded_shard_length:
            The size to which each shard list is padded.

        :return padded_negatives: shape: (n_triple, n_shard, padded_shard_length)
            The padded shard lists of negatives for each triple.
        :return mask: shape: (n_triple, n_shard, padded_shard_length)
            Mask for ture entries in `padded_negatives` (False for padding enries).
        :rtype: _type_
        """
        mask = np.arange(padded_shard_length)[None, None, :] < shard_counts[..., None]

        shard_idx = (
            np.arange(padded_shard_length)[None, None, :] % shard_counts[..., None]
        )
        global_idx = np.minimum(shard_idx + shard_offsets[..., None], self.nss - 1)

        padded_negatives = negatives[
            np.arange(self.n_triple)[:, None, None], global_idx
        ]

        return padded_negatives, mask
