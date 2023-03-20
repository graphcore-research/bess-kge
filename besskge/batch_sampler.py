# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import einops
import numpy as np
import torch

from besskge.negative_sampler import ShardedNegativeSampler
from besskge.sharding import ShardedTripleSet


class ShardedBatchSampler(torch.utils.data.Dataset, ABC):
    """
    Base class for sharded batch sampler.
    """

    def __init__(
        self,
        sharded_triple_set: ShardedTripleSet,
        negative_sampler: ShardedNegativeSampler,
        shard_bs: int,
        batches_per_step: int,
        seed: int,
        hrt_freq_weighting: bool = False,
        weight_smoothing: float = 0.0,
        duplicate_batch: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize sharded batch sampler.

        :param part:
            The part of :attr:`dataset` to sample from.
        :param dataset:
            The KG dataset.
        :param sharding:
            The entity sharding.
        :param negative_sampler:
            The sampler of negative entities.
        :param shard_bs:
            The microbatch size, i.e. the number of positive triples
            processed on each shard.
        :param batches_per_step:
            The number of batches to sample at each call.
        :param seed:
            The RNG seed.
        :param hrt_freq_weighting:
            Use frequency-based triple weighting as in [...], defaults to False.
        :param weight_smoothing:
            Weight-smoothing parameter for frequency-based
            triple weigthing, defaults to 0.0.
        :param duplicate_batch:
            The batch sampled from each shardpair has two identical halves;
            to be used with "ht" corruption scheme. Defaults to False.
        """
        self.n_shard = sharded_triple_set.sharding.n_shard
        self.triples = sharded_triple_set.sharded_triples
        self.shardpair_counts = sharded_triple_set.shardpair_counts
        self.shardpair_offsets = sharded_triple_set.shardpair_offsets

        self.shard_bs = shard_bs
        self.duplicate_batch = duplicate_batch
        # Each microbatch is formed of n_shard blocks, based on tail location
        self.positive_per_shardpair = int(np.ceil(self.shard_bs / self.n_shard))
        if self.duplicate_batch:
            self.positive_per_shardpair //= 2
        self.batches_per_step = batches_per_step
        self.hrt_freq_weighting = hrt_freq_weighting
        self.shardpair_sample_size = self.batches_per_step * self.positive_per_shardpair
        self.negative_sampler = negative_sampler
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        if self.hrt_freq_weighting:
            _, hr_idx, hr_count = np.unique(
                self.triples[..., 0]
                + sharded_triple_set.sharding.n_entity * self.triples[..., 1],
                return_counts=True,
                return_inverse=True,
            )
            _, rt_idx, rt_count = np.unique(
                self.triples[..., 2]
                + sharded_triple_set.sharding.n_entity * self.triples[..., 1],
                return_counts=True,
                return_inverse=True,
            )
            self.hrt_weights = np.sqrt(
                1.0 / (hr_count[hr_idx] + rt_count[rt_idx] + weight_smoothing)
            )

    def __len__(self) -> int:
        """
        The length of the batch sampler is based on the length of
        the largest shardpair.
        At each call, :attr:`ShardedBatchSampler.shardpair_sample_size` triples
        for each shardpair are returned.

        :return:
            The length of the batch sampler.
        """
        return (
            int(np.ceil(self.shardpair_counts.max() / self.shardpair_sample_size))
            * self.shardpair_sample_size
        )

    def __getitem__(self, idx: List[int]) -> Dict[str, np.ndarray]:
        """
        Sample batch.

        :param idx:
            The index of the batch.

        :return:
            Indices of head, relation, tail and negative entities in the batch,
            and associated weights / masks.
        """
        sample_triple_dict = self.sample_triples(idx)
        if self.duplicate_batch:
            sample_triple_dict = {
                k: einops.repeat(
                    v,
                    "step shard_h shard_t triple -> step shard_h shard_t (2 triple)",
                )
                for k, v in sample_triple_dict.items()
            }
        sample_idx = sample_triple_dict.pop("sample_idx")
        head, relation, tail = einops.rearrange(
            self.triples[sample_idx],
            "step shard_h shard_t triple hrt -> hrt step shard_h shard_t triple",
        )
        tail = einops.rearrange(
            tail, "step shard_h shard_t triple -> step shard_t shard_h triple"
        )

        sample_negative_dict = self.negative_sampler.get_negative_batch(sample_idx)
        negative_entities = sample_negative_dict.pop("negative_entities")

        if self.hrt_freq_weighting:
            triple_weight = einops.rearrange(
                self.hrt_weights[sample_idx],
                "step shard_h shard_t triple -> step shard_h (shard_t triple)",
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
    def sample_triples(self, idx: List[int]) -> Dict[str, np.ndarray]:
        """
        Sample positive triples in the batch.

        :param idx:
            The index of the batch.
        """
        raise NotImplementedError

    def get_dataloader_sampler(self, shuffle: bool) -> torch.utils.data.Sampler:
        """
        Instantiate appropriate :class:`torch.data.Sampler` for the
        :class:`torch.utils.data.DataLoader` to be used with the
        sharded batch sampler.

        :param shuffle:
            Shuffle triples at each new epoch.
        :return:
            The dataloader sampler.
        """
        sampler = (
            torch.utils.data.RandomSampler(self)
            if shuffle
            else torch.utils.data.SequentialSampler(self)
        )
        return torch.utils.data.BatchSampler(
            sampler, batch_size=self.shardpair_sample_size, drop_last=False
        )

    def get_dataloader(
        self,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Instantiate  appropriate :class:`torch.utils.data.DataLoader`
        to iterate over the batch sampler.

        :param shuffle:
            Shuffle triples at each new epoch, defaults to True.
        :param num_workers:
            see :meth:`torch.utils.data.DataLoader.__init__`, defaults to 0.
        :param persistent_workers:
            see :meth:`torch.utils.data.DataLoader.__init__`, defaults to False.
        :return:
            The dataloader.
        """
        dl_sampler = self.get_dataloader_sampler(shuffle=shuffle)
        return torch.utils.data.DataLoader(
            self,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            worker_init_fn=self.worker_init_fn,
            sampler=dl_sampler,
        )

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """
        Worker intialization function to be passed to
        :class:`torch.utils.data.DataLoader`.

        :param worker_id:
            Worker ID.
        """
        worker_info = torch.utils.data.get_worker_info()
        dataset_unwrapped = worker_info.dataset
        worker_seed = dataset_unwrapped.seed + worker_id
        dataset_unwrapped.rng = np.random.RandomState(worker_seed)
        dataset_unwrapped.negative_sampler.rng = np.random.RandomState(worker_seed)


class RigidShardedBatchSampler(ShardedBatchSampler):
    """
    At each call, sample same indices from all shardpairs, repeating triples
    in shorter ones to pad to same length. Returns mask to identify
    padding triples.
    """

    # docstr-coverage: inherited
    def __init__(
        self,
        sharded_triples: ShardedTripleSet,
        negative_sampler: ShardedNegativeSampler,
        shard_bs: int,
        batches_per_step: int,
        seed: int,
        hrt_freq_weighting: bool = False,
        weight_smoothing: float = 0.0,
        duplicate_batch: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super(RigidShardedBatchSampler, self).__init__(
            sharded_triples,
            negative_sampler,
            shard_bs,
            batches_per_step,
            seed,
            hrt_freq_weighting,
            weight_smoothing,
            duplicate_batch,
            *args,
            **kwargs,
        )

        padded_shardpair_length = len(self)
        self.triple_mask = (
            np.arange(padded_shardpair_length)[None, None, :]
            < self.shardpair_counts[..., None]
        )  # shape (n_shard, n_shard, padded_shardpair_length)
        triple_padded_idx = (
            np.arange(padded_shardpair_length)[None, None, :]
            % self.shardpair_counts[..., None]
        )
        self.triple_padded_idx = np.minimum(
            triple_padded_idx + self.shardpair_offsets[..., None],
            np.cumsum(self.shardpair_counts.sum(-1))[:, None, None],
        )  # shape (n_shard, n_shard, padded_shardpair_length)

    # docstr-coverage: inherited
    def sample_triples(self, idx: List[int]) -> Dict[str, np.ndarray]:
        sample_idx = einops.rearrange(
            self.triple_padded_idx[:, :, idx],
            "shard_h shard_t (step triple) -> step shard_h shard_t triple",
            step=self.batches_per_step,
        )
        batch_mask = einops.rearrange(
            self.triple_mask[:, :, idx],
            "shard_h shard_t (step triple) -> step shard_h shard_t triple",
            step=self.batches_per_step,
        )

        return dict(sample_idx=sample_idx, triple_mask=batch_mask)


class RandomShardedBatchSampler(ShardedBatchSampler):
    """
    Sample random indices (with replacement) from all shardpairs.
    No padding of shardpairs applied.
    """

    # docstr-coverage: inherited
    def sample_triples(self, idx: List[int]) -> Dict[str, np.ndarray]:
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

    def __len__(self) -> int:
        return int(np.ceil(self.shardpair_counts.max() / self.shardpair_sample_size))

    # docstr-coverage: inherited
    def get_dataloader_sampler(self, shuffle=False) -> torch.utils.data.Sampler:
        return torch.utils.data.SequentialSampler(self)
