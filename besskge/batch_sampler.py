# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Union, cast

import einops
import numpy as np
import poptorch
import torch
from numpy.typing import NDArray

from besskge.negative_sampler import ShardedNegativeSampler
from besskge.sharding import PartitionedTripleSet


class ShardedBatchSampler(torch.utils.data.Dataset[Dict[str, torch.Tensor]], ABC):
    """
    Base class for sharded batch sampler.
    """

    def __init__(
        self,
        partitioned_triple_set: PartitionedTripleSet,
        negative_sampler: ShardedNegativeSampler,
        shard_bs: int,
        batches_per_step: int,
        seed: int,
        hrt_freq_weighting: bool = False,
        weight_smoothing: float = 0.0,
        duplicate_batch: bool = False,
        return_triple_idx: bool = False,
    ):
        """
        Initialize sharded batch sampler.

        :param partitioned_triple_set:
            The pre-processed collection of triples.
        :param negative_sampler:
            The sampler for negative entities.
        :param shard_bs:
            The micro-batch size. This is the number of positive triples
            processed on each shard.
        :param batches_per_step:
            The number of batches to sample at each call.
        :param seed:
            The RNG seed.
        :param hrt_freq_weighting:
            If True, uses frequency-based triple weighting. Default: False.
        :param weight_smoothing:
            Weight-smoothing parameter for frequency-based
            triple weighting. Default: 0.0.
        :param duplicate_batch:
            If True, the batch sampled from each triple partition has two
            identical halves.
            This is to be used with "ht" corruption scheme at inference time.
            Default: False.
        :param return_triple_idx:
            If True, return the indices (wrt partitioned_triple_set.triples)
            of the triples in the batch. Default: False.
        """
        self.n_shard = partitioned_triple_set.sharding.n_shard
        self.triples = partitioned_triple_set.triples
        self.dummy = partitioned_triple_set.dummy
        self.triple_counts = partitioned_triple_set.triple_counts
        self.triple_offsets = partitioned_triple_set.triple_offsets
        self.triple_partition_mode = partitioned_triple_set.partition_mode
        self.negative_sampler = negative_sampler

        self.shard_bs = shard_bs
        self.batches_per_step = batches_per_step
        self.duplicate_batch = duplicate_batch
        if self.triple_partition_mode == "ht_shardpair":
            # The micro-batch on device N is formed of n_shard blocks,
            # corresponding to triple partitions (h_shard, t_shard)
            # with h_shard = N and t_shard = 0, ..., n_shard-1.
            self.positive_per_partition = int(np.ceil(self.shard_bs / self.n_shard))
        else:
            self.positive_per_partition = self.shard_bs
        if self.duplicate_batch:
            self.positive_per_partition //= 2
        if self.negative_sampler.corruption_scheme == "ht":
            # Each partition is split into two halves, so we need positive_per_partition
            # to be even.
            self.positive_per_partition = (self.positive_per_partition // 2) * 2

        # Total number of triples sampled from each partition at each call
        self.partition_sample_size = self.batches_per_step * self.positive_per_partition

        self.hrt_freq_weighting = hrt_freq_weighting
        self.return_triple_idx = return_triple_idx
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        if self.hrt_freq_weighting:
            if self.dummy != "none":
                warnings.warn(
                    "hrt frequency weights are being computed on dummy entities"
                )
            _, hr_idx, hr_count = np.unique(
                self.triples[..., 0]
                + partitioned_triple_set.sharding.n_entity * self.triples[..., 1],
                return_counts=True,
                return_inverse=True,
            )
            _, rt_idx, rt_count = np.unique(
                self.triples[..., 2]
                + partitioned_triple_set.sharding.n_entity * self.triples[..., 1],
                return_counts=True,
                return_inverse=True,
            )
            self.hrt_weights = np.sqrt(
                1.0 / (hr_count[hr_idx] + rt_count[rt_idx] + weight_smoothing)
            )

    def __len__(self) -> int:
        """
        Returns the length of the batch sampler.

        The length of the batch sampler is based on the length of
        the largest triple partition.
        At each call, :attr:`ShardedBatchSampler.partition_sample_size` triples
        for each partition are returned.

        :return:
            The length of the batch sampler.
        """
        return (
            int(np.ceil(self.triple_counts.max() / self.partition_sample_size))
            * self.partition_sample_size
        )

    def __getitem__(self, idx: List[int]) -> Dict[str, torch.Tensor]:
        """
        Sample batch.

        :param idx:
            The batch index.

        :return:
            Indices of head, relation, tail and negative entities in the batch,
            and associated weights and masks.
        """
        sample_triple_dict = self.sample_triples(idx)
        if self.duplicate_batch:
            sample_triple_dict = {
                k: einops.repeat(
                    v,
                    "step shard ... triple -> step shard ... (2 triple)",
                )
                for k, v in sample_triple_dict.items()
            }
        sample_idx = cast(NDArray[np.int64], sample_triple_dict.pop("sample_idx"))
        head, relation, tail = einops.rearrange(
            self.triples[sample_idx],
            "... hrt -> hrt ...",
        )
        if self.triple_partition_mode == "ht_shardpair":
            # Prepare tail indices for AllToAll exchange
            tail = einops.rearrange(
                tail, "step shard_h shard_t triple -> step shard_t shard_h triple"
            )

        batch_dict = {
            "head": head.astype(np.int32),
            "relation": relation.astype(np.int32),
            "tail": tail.astype(np.int32),
            **sample_triple_dict,
        }
        sample_negative_dict = self.negative_sampler(sample_idx)
        if "negative_entities" in sample_negative_dict.keys():
            negative_entities = sample_negative_dict.pop("negative_entities")
            batch_dict.update(negative=negative_entities.astype(np.int32))
        batch_dict.update(**sample_negative_dict)

        if self.dummy in ["head", "tail"]:
            batch_dict.pop(self.dummy)

        if self.hrt_freq_weighting:
            triple_weight = einops.rearrange(
                self.hrt_weights[sample_idx],
                "step shard ... triple -> step shard (... triple)",
            )
            triple_weight /= np.sum(triple_weight, axis=-1, keepdims=True)
            triple_weight *= self.shard_bs
            batch_dict.update(triple_weight=triple_weight.astype(np.float32))

        if self.return_triple_idx:
            batch_dict.update(triple_idx=sample_idx)

        return {k: torch.from_numpy(v) for k, v in batch_dict.items()}

    @abstractmethod
    def sample_triples(
        self, idx: List[int]
    ) -> Dict[str, Union[NDArray[np.int64], NDArray[np.bool_]]]:
        """
        Sample positive triples in the batch.

        :param idx:
            The batch index.
        :return:
            Per-partition indices of positive triples, and other relevant data.
        """
        raise NotImplementedError

    def get_dataloader_sampler(
        self, shuffle: bool
    ) -> torch.utils.data.Sampler[List[int]]:
        """
        Returns the dataloader sampler.

        Instantiate the appropriate :class:`torch.data.Sampler` class for the
        :class:`torch.utils.data.DataLoader` class to be used with the
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
            sampler, batch_size=self.partition_sample_size, drop_last=False
        )

    def get_dataloader(
        self,
        options: poptorch.Options,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
        buffer_size: int = 16,
    ) -> poptorch.DataLoader:
        """
        Returns the PopTorch dataloader.

        Instantiate the appropriate :class:`poptorch.DataLoader`
        class to iterate over the batch sampler.
        It uses asynchronous data-loading to minimize CPU-IPU I/O.

        :param options:
            `poptorch.Options` used to compile and run the model.
        :param shuffle:
            If True, shuffles triples at each new epoch. Default: True.
        :param num_workers:
            see :meth:`torch.utils.data.DataLoader.__init__`. Default: 0.
        :param persistent_workers:
            see :meth:`torch.utils.data.DataLoader.__init__`. Default: False.
        :param buffer_size:
            Size of the ring buffer in shared memory used to preload batches.
        :return:
            The PopTorch dataloader.
        """

        return poptorch.DataLoader(
            options=options,
            dataset=self,
            batch_size=None,
            sampler=self.get_dataloader_sampler(shuffle=shuffle),
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            worker_init_fn=self.worker_init_fn,
            mode=poptorch.DataLoaderMode.Async,
            async_options={
                "early_preload": True,
                "buffer_size": buffer_size,
                "sharing_strategy": poptorch.SharingStrategy.SharedMemory,
            },
        )

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """
        Worker initialization function to be passed to
        :class:`torch.utils.data.DataLoader`.

        :param worker_id:
            Worker ID.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            dataset_unwrapped = cast(ShardedBatchSampler, worker_info.dataset)
            worker_seed = dataset_unwrapped.seed + worker_id
            dataset_unwrapped.rng = np.random.default_rng(worker_seed)
            dataset_unwrapped.negative_sampler.rng = np.random.default_rng(worker_seed)


class RigidShardedBatchSampler(ShardedBatchSampler):
    """
    At each call, sample triples with the same specified indices from all
    triple partitions,
    repeating triples in shorter ones to pad to the same length.
    Returns a mask to identify padding triples.
    """

    # docstr-coverage: inherited
    def __init__(
        self,
        partitioned_triple_set: PartitionedTripleSet,
        negative_sampler: ShardedNegativeSampler,
        shard_bs: int,
        batches_per_step: int,
        seed: int,
        hrt_freq_weighting: bool = False,
        weight_smoothing: float = 0.0,
        duplicate_batch: bool = False,
        return_triple_idx: bool = False,
    ) -> None:
        super(RigidShardedBatchSampler, self).__init__(
            partitioned_triple_set,
            negative_sampler,
            shard_bs,
            batches_per_step,
            seed,
            hrt_freq_weighting,
            weight_smoothing,
            duplicate_batch,
            return_triple_idx,
        )

        padded_partition_length = len(self)
        expand_axes = (0, 1) if self.triple_partition_mode == "ht_shardpair" else (0,)
        self.triple_mask = (
            np.expand_dims(np.arange(padded_partition_length), axis=expand_axes)
            < self.triple_counts[..., None]
        )  # shape (n_shard, [n_shard,] padded_partition_length)
        triple_padded_idx = (
            np.expand_dims(np.arange(padded_partition_length), axis=expand_axes)
            % self.triple_counts[..., None]
        ) + self.triple_offsets[..., None]
        # Index safeguard for when the last partition is empty
        self.triple_padded_idx = np.minimum(
            triple_padded_idx,
            self.triples.shape[0] - 1,
        )  # shape (n_shard, [n_shard,] padded_partition_length)

    # docstr-coverage: inherited
    def sample_triples(
        self, idx: List[int]
    ) -> Dict[str, Union[NDArray[np.int64], NDArray[np.bool_]]]:
        sample_idx = einops.rearrange(
            self.triple_padded_idx[..., idx],
            "shard ... (step triple) -> step shard ... triple",
            step=self.batches_per_step,
        )
        batch_mask = einops.rearrange(
            self.triple_mask[..., idx],
            "shard ... (step triple) -> step shard ... triple",
            step=self.batches_per_step,
        )

        return dict(sample_idx=sample_idx, triple_mask=batch_mask)


class RandomShardedBatchSampler(ShardedBatchSampler):
    """
    Sample random indices (with replacement) from all triple partitions.
    No padding of triple partitions applied.
    """

    # docstr-coverage: inherited
    def sample_triples(
        self, idx: List[int]
    ) -> Dict[str, Union[NDArray[np.int64], NDArray[np.bool_]]]:
        sample_size = (
            (
                self.batches_per_step,
                self.n_shard,
                self.n_shard,
                self.positive_per_partition,
            )
            if self.triple_partition_mode == "ht_shardpair"
            else (
                self.batches_per_step,
                self.n_shard,
                self.positive_per_partition,
            )
        )

        sample_idx = np.expand_dims(
            self.triple_offsets, axis=(0, -1)
        ) + self.rng.integers(
            1 << 63,
            size=sample_size,
        ) % np.expand_dims(
            self.triple_counts, axis=(0, -1)
        )
        return dict(sample_idx=sample_idx)

    def __len__(self) -> int:
        return int(np.ceil(self.triple_counts.max() / self.partition_sample_size))

    # docstr-coverage: inherited
    def get_dataloader_sampler(
        self, shuffle: bool = True
    ) -> torch.utils.data.Sampler[List[int]]:
        sampler = torch.utils.data.SequentialSampler(self)
        return torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)
