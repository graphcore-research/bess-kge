import numpy as np
from abc import ABC, abstractmethod

from besskge.dataset import Sharding


class ShardedNegativeSampler(ABC):
    @abstractmethod
    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class RandomShardedNegativeSampler(ShardedNegativeSampler):
    def __init__(
        self,
        n_negative: int,
        sharding: Sharding,
        corruption_scheme: str,
        local_sampling: bool,
        seed: int,
        flat_negative_format: bool = False,
        *args,
        **kwargs,
    ):
        self.n_negative = n_negative
        self.sharding = sharding
        self.shard_counts = sharding.shard_counts
        self.corruption_scheme = corruption_scheme
        self.local_sampling = local_sampling
        self.rng = np.random.RandomState(seed=seed)
        self.flat_negative_format = flat_negative_format

    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> np.ndarray:
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
    ):
        super(TypeBasedShardedNegativeSampler, self).__init__(
            n_negative,
            sharding,
            corruption_scheme,
            local_sampling,
            seed,
            *args,
            **kwargs,
        )
        self.triple_types = triple_types
        self.type_offsets = sharding.entity_type_offsets
        self.type_counts = sharding.entity_type_counts

    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> np.ndarray:
        batches_per_step, n_shard, _, positive_per_shardpair = sample_idx.shape
        head_type, tail_type = self.triple_types[sample_idx].transpose(4, 0, 1, 2, 3)

        if self.corruption_scheme == "h":
            relevant_type = head_type.reshape(batches_per_step, n_shard, 1, -1)
        elif self.corruption_scheme == "t":
            relevant_type = tail_type.reshape(batches_per_step, n_shard, 1, -1)
        elif self.corruption_scheme == "ht":
            # cut_point = (n_shard * positive_per_shardpair) // 2
            # head_type_batch = head_type.reshape(*head_type.shape[:-2], 1, -1)
            # tail_type_batch = tail_type.reshape(*tail_type.shape[:-2], 1, -1)
            # relevant_type = np.concatenate([head_type_batch[...,:cut_point], tail_type_batch[...,cut_point:]], axis=-1)
            cut_point = positive_per_shardpair // 2
            relevant_type = np.concatenate(
                [head_type[..., :cut_point], tail_type[..., cut_point:]], axis=-1
            ).reshape(batches_per_step, n_shard, 1, -1)
        else:
            raise NotImplementedError(
                f"Corruption scheme {self.corruption_scheme} not supported by {self.__class__}"
            )

        relevant_type = np.repeat(relevant_type, n_shard, axis=-2)
        if not self.local_sampling:
            relevant_type = relevant_type.transpose(0, 2, 1, 3)

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


class EntityBasedShardedNegativeSampler(ShardedNegativeSampler):
    def __init__(
        self,
        negative_heads: np.ndarray,
        negative_tails: np.ndarray,
        max_negative_per_step: int,
        sharding: Sharding,
        corruption_scheme: str,
        seed: int,
        *args,
        **kwargs,
    ):
        if (
            negative_heads is not None
            and negative_tails is not None
            and negative_heads.shape != negative_tails.shape
        ):
            raise ValueError("negative heads and tails need to have the same dimension")
        self.n_triple, self.nss = negative_heads.shape
        self.max_negative_per_step = max_negative_per_step
        self.sharding = sharding
        self.shard_counts = sharding.shard_counts
        self.corruption_scheme = corruption_scheme
        self.local_sampling = False
        self.flat_negative_format = False
        self.rng = np.random.RandomState(seed=seed)

        if self.corruption_scheme in ["h", "t"]:
            negatives = (
                negative_heads if self.corruption_scheme == "h" else negative_tails
            )
            (
                shard_neg_counts,
                shard_neg_offsets,
                sort_neg_idx,
            ) = self.shard_negatives(negatives)
            self.shard_padding_length = shard_neg_counts.max()
            self.padded_negatives, self.mask = self.pad_negatives(
                sharding.entity_to_idx[
                    negatives[np.arange(self.n_triple)[:, None], sort_neg_idx]
                ],
                shard_neg_counts,
                shard_neg_offsets,
                self.shard_padding_length,
            )
        elif self.corruption_scheme == "ht":
            (
                shard_neg_h_counts,
                shard_neg_h_offsets,
                sort_neg_h_idx,
            ) = self.shard_negatives(negative_heads)
            (
                shard_neg_t_counts,
                shard_neg_t_offsets,
                sort_neg_t_idx,
            ) = self.shard_negatives(negative_tails)
            self.shard_padding_length = np.max(
                [shard_neg_h_counts.max(), shard_neg_t_counts.max()]
            )
            self.padded_negatives_h, self.mask_h = self.pad_negatives(
                sharding.entity_to_idx[
                    negative_heads[np.arange(self.n_triple)[:, None], sort_neg_h_idx]
                ],
                shard_neg_h_counts,
                shard_neg_h_offsets,
                self.shard_padding_length,
            )
            self.padded_negatives_t, self.mask_t = self.pad_negatives(
                sharding.entity_to_idx[
                    negative_tails[np.arange(self.n_triple)[:, None], sort_neg_t_idx]
                ],
                shard_neg_t_counts,
                shard_neg_t_offsets,
                self.shard_padding_length,
            )
        else:
            raise NotImplementedError(
                f"Corruption scheme {self.corruption_scheme} not supported by {self.__class__}"
            )

    def get_negative_batch(
        self,
        sample_idx: np.ndarray,
    ) -> np.ndarray:
        batches_per_step, n_shard, _, positive_per_shardpair = sample_idx.shape
        if self.corruption_scheme in ["h", "t"]:
            sample_idx = sample_idx.reshape(batches_per_step, n_shard, -1)
            negative_entities = self.padded_negatives[sample_idx].transpose(
                0, 3, 1, 2, 4
            )
            negative_mask = self.mask[sample_idx].transpose(0, 3, 1, 2, 4)
        elif self.corruption_scheme == "ht":
            cutpoint = positive_per_shardpair // 2
            negative_entities = (
                np.concatenate(
                    [
                        self.padded_negatives_h[sample_idx][:, :, :, :cutpoint],
                        self.padded_negatives_t[sample_idx][:, :, :, cutpoint:],
                    ],
                    axis=3,
                )
                .reshape(
                    batches_per_step, n_shard, -1, n_shard, self.shard_padding_length
                )
                .transpose(0, 3, 1, 2, 4)
            )
            negative_mask = (
                np.concatenate(
                    [
                        self.mask_h[sample_idx][:, :, :, :cutpoint],
                        self.mask_t[sample_idx][:, :, :, cutpoint:],
                    ],
                    axis=3,
                )
                # .reshape(
                #     batches_per_step, n_shard, -1, n_shard, self.shard_padding_length
                # )
                # .transpose(0, 3, 1, 2, 4)
                .reshape(
                    batches_per_step, n_shard, -1, n_shard * self.shard_padding_length
                )
            )

        return dict(negative_entities=negative_entities, negative_mask=negative_mask)

    def shard_negatives(
        self,
        negatives: np.ndarray,
    ) -> np.ndarray:
        n_shard = self.sharding.n_shard
        shard_neg = self.sharding.entity_to_shard[negatives]
        shard_neg_counts = np.bincount(
            (shard_neg + n_shard * np.arange(self.n_triple)[:, None]).flatten(),
            minlength=n_shard * self.n_triple,
        ).reshape(self.n_triple, n_shard)
        shard_neg_offsets = np.c_[
            [0] * self.n_triple, np.cumsum(shard_neg_counts, axis=-1)[:, :-1]
        ]
        sort_neg_idx = np.argsort(shard_neg, axis=-1)

        return shard_neg_counts, shard_neg_offsets, sort_neg_idx

    def pad_negatives(
        self,
        negatives: np.ndarray,
        shard_counts: np.ndarray,
        shard_offsets: np.ndarray,
        shard_padding_length: np.ndarray,
    ) -> np.ndarray:
        shard_idx = (
            np.arange(shard_padding_length)[None, None, :] % shard_counts[..., None]
        )

        global_idx = np.minimum(shard_idx + shard_offsets[..., None], self.nss - 1)

        padded_negatives = negatives[
            np.arange(self.n_triple)[:, None, None], global_idx
        ]

        mask = np.arange(shard_padding_length)[None, None, :] < shard_counts[..., None]

        return padded_negatives, mask
