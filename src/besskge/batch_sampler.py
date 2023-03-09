import numpy as np
import torch
from abc import ABC, abstractmethod

from besskge.dataset import KGDataset, Sharding
from besskge.negative_sampler import (
    ShardedNegativeSampler,
    TypeBasedShardedNegativeSampler,
    EntityBasedShardedNegativeSampler,
)


class ShardedBatchSampler(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        part: str,
        dataset: KGDataset,
        sharding: Sharding,
        negative_sampler: ShardedNegativeSampler,
        shard_bs: int,
        batches_per_step: int,
        seed: int,
        hrt_freq_weighting: bool = False,
        weight_smoothing: float = 0.0,
        *args,
        **kwargs,
    ):
        self.n_shard = sharding.n_shard
        self.shard_bs = shard_bs
        # Each microbatch is formed of n_shard blocks, based on tail location
        self.positive_per_shardpair = int(np.ceil(self.shard_bs / self.n_shard))
        self.batches_per_step = batches_per_step
        self.hrt_freq_weighting = hrt_freq_weighting
        self.shardpair_sample_size = self.batches_per_step * self.positive_per_shardpair
        self.negative_sampler = negative_sampler
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        (
            self.triples,
            self.shardpair_counts,
            self.shardpair_offsets,
            self.sort_idx,
        ) = self.shard_triples(dataset, part, sharding)

        # if isinstance(self.negative_sampler, TypeBasedShardedNegativeSampler):
        #     self.negative_sampler.triple_types = self.negative_sampler.triple_types[
        #         sort_idx
        #     ]
        # if isinstance(self.negative_sampler, EntityBasedShardedNegativeSampler):
        #     self.negative_sampler.negative_heads = self.negative_sampler.negative_heads[
        #         sort_idx
        #     ]
        #     self.negative_sampler.negative_tails = self.negative_sampler.negative_tails[
        #         sort_idx
        #     ]

        if self.hrt_freq_weighting:
            _, hr_idx, hr_count = np.unique(
                self.triples[..., 0] + dataset.n_entity * self.triples[..., 1],
                return_counts=True,
                return_inverse=True,
            )
            _, rt_idx, rt_count = np.unique(
                self.triples[..., 2] + dataset.n_entity * self.triples[..., 1],
                return_counts=True,
                return_inverse=True,
            )
            self.hrt_weights = np.sqrt(
                1.0 / (hr_count[hr_idx] + rt_count[rt_idx] + weight_smoothing)
            )

    def __len__(self):
        return int(np.ceil(self.shardpair_counts.max() / self.shardpair_sample_size))

    def __getitem__(self, idx):
        sample_triple_dict = self.get_sample_triple_idx(idx)
        sample_idx = sample_triple_dict.pop("sample_idx")
        head, relation, tail = self.triples[sample_idx].transpose(4, 0, 1, 2, 3)
        tail = tail.transpose(0, 2, 1, 3)

        sample_negative_dict = self.negative_sampler.get_negative_batch(sample_idx)
        negative_entities = sample_negative_dict.pop("negative_entities")

        if self.hrt_freq_weighting:
            triple_weight = self.hrt_weights[sample_idx].reshape(
                *sample_idx.shape[:-2], -1
            )
            triple_weight /= np.sum(triple_weight, axis=-1, keepdims=True)
        else:
            triple_weight = (
                np.ones((self.batches_per_step, self.n_shard, 1)) / self.shard_bs
            )

        return {
            "head": head.astype(np.int32),
            "relation": relation.astype(np.int32),
            "tail": tail.astype(np.int32),
            "negative": negative_entities.astype(np.int32),
            "triple_weight": triple_weight.astype(np.float32),
            **sample_triple_dict,
            **sample_negative_dict,
        }

    @abstractmethod
    def get_sample_triple_idx(self, idx):
        raise NotImplementedError

    @classmethod
    def shard_triples(
        self, dataset: KGDataset, part: str, sharding: Sharding
    ) -> np.ndarray:
        """Divide triples in sharding.n_shard**2 buckets, based on head and tail shards."""
        triples = dataset.triples[part]
        n_shard = sharding.n_shard
        shard_h, shard_t = sharding.entity_to_shard[triples[:, [0, 2]].T]
        shardpair_idx = shard_h * n_shard + shard_t
        shardpair_counts = np.bincount(
            shardpair_idx, minlength=n_shard * n_shard
        ).reshape(n_shard, n_shard)
        shardpair_offsets = np.concatenate(
            [[0], np.cumsum(shardpair_counts)[:-1]]
        ).reshape(n_shard, n_shard)
        sort_idx = np.argsort(shardpair_idx)
        sharded_triples = triples[sort_idx]
        sharded_triples[:, [0, 2]] = sharding.entity_to_idx[sharded_triples[:, [0, 2]]]
        return sharded_triples, shardpair_counts, shardpair_offsets, sort_idx

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset_unwrapped = worker_info.dataset
        worker_seed = dataset_unwrapped.seed + worker_id
        dataset_unwrapped.rng = np.random.RandomState(worker_seed)
        dataset_unwrapped.negative_sampler.rng = np.random.RandomState(worker_seed)


class RandomShardedBatchSampler(ShardedBatchSampler):
    def get_sample_triple_idx(self, idx):
        sample_idx = (
            self.shardpair_offsets[None, :, :, None]
            + self.rng.randint(
                1 << 63,
                size=(
                    self.batches_per_step,
                    self.n_shard,
                    self.n_shard,
                    self.positive_per_shardpair,
                ),
            )
            % self.shardpair_counts[None, :, :, None]
        )
        return dict(sample_idx=sample_idx)


class SequentialShardedBatchSampler(ShardedBatchSampler):
    def __init__(
        self,
        part: str,
        dataset: KGDataset,
        sharding: Sharding,
        negative_sampler: ShardedNegativeSampler,
        shard_bs: int,
        batches_per_step: int,
        seed: int,
        hrt_freq_weighting: bool = False,
        weight_smoothing: float = 0.0,
        *args,
        **kwargs,
    ):
        super(SequentialShardedBatchSampler, self).__init__(
            part,
            dataset,
            sharding,
            negative_sampler,
            shard_bs,
            batches_per_step,
            seed,
            hrt_freq_weighting,
            weight_smoothing,
            *args,
            **kwargs,
        )

        self.padded_shardpair_length = self.shardpair_sample_size * len(self)
        self.triple_mask = (
            np.arange(self.padded_shardpair_length)[None, None, :]
            < self.shardpair_counts[..., None]
        )
        triple_padded_idx = (
            np.arange(self.padded_shardpair_length)[None, None, :]
            % self.shardpair_counts[..., None]
        )
        self.triple_padded_idx = np.minimum(
            triple_padded_idx + self.shardpair_offsets[..., None],
            np.cumsum(self.shardpair_counts.sum(-1))[:, None, None],
        )

    def get_sample_triple_idx(self, idx):
        offset = self.shardpair_sample_size * idx

        sample_idx = (
            self.triple_padded_idx[:, :, offset : offset + self.shardpair_sample_size]
            .reshape(self.n_shard, self.n_shard, self.batches_per_step, -1)
            .transpose(2, 0, 1, 3)
        )
        batch_mask = (
            self.triple_mask[:, :, offset : offset + self.shardpair_sample_size]
            .reshape(self.n_shard, self.n_shard, self.batches_per_step, -1)
            .transpose(2, 0, 1, 3)
        )

        return dict(sample_idx=sample_idx, triple_mask=batch_mask)

    def shuffle_triples(self):
        # self.seed += 100
        # self.rng = np.random.RandomState(self.seed)
        self.triple_padded_idx = self.triple_padded_idx[
            np.arange(self.n_shard)[:, None, None],
            np.arange(self.n_shard)[None, :, None],
            self.rng.permutation(self.padded_shardpair_length),
        ]
