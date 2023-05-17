# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import einops
import numpy as np
import pytest
from numpy.testing import assert_equal

from besskge.negative_sampler import (
    RandomShardedNegativeSampler,
    TripleBasedShardedNegativeSampler,
    TypeBasedShardedNegativeSampler,
)
from besskge.sharding import Sharding

seed = 1234
n_entity = 500
n_shard = 4
n_triple = 2000
batches_per_step = 5
positive_per_partition = 60
cutpoint = positive_per_partition // 2
n_negative = 250

sample_idx_size = {
    "shard": (batches_per_step, n_shard, positive_per_partition),
    "shardpair": (batches_per_step, n_shard, n_shard, positive_per_partition),
}


@pytest.mark.parametrize("triple_partition_mode", ["shard", "shardpair"])
@pytest.mark.parametrize("flat_negative_format", [True, False])
def test_random_sharded_ns(
    triple_partition_mode: str, flat_negative_format: bool
) -> None:
    np.random.seed(seed)
    sharding = Sharding.create(n_entity, n_shard, seed=seed)

    ns = RandomShardedNegativeSampler(
        n_negative=n_negative,
        sharding=sharding,
        seed=seed,
        corruption_scheme="ht",
        local_sampling=False,
        flat_negative_format=flat_negative_format,
    )

    # Negative batch is independent on sample_idx, corruption_scheme and local_sampling
    sample_idx = np.ones(sample_idx_size[triple_partition_mode], dtype=np.int64)
    neg_batch = ns(sample_idx)["negative_entities"]

    # Check sampled entities idx are smaller than number of actual entitites in the shard
    for processing_shard in range(n_shard):
        assert (
            neg_batch[:, processing_shard].max()
            < sharding.shard_counts[processing_shard]
        )


@pytest.mark.parametrize("triple_partition_mode", ["shard", "shardpair"])
@pytest.mark.parametrize("local_sampling", [True, False])
@pytest.mark.parametrize("corruption_scheme", ["h", "t", "ht"])
def test_type_based_sharded_ns(
    triple_partition_mode: str, local_sampling: bool, corruption_scheme: str
) -> None:
    np.random.seed(seed)
    n_type = 3
    # Type IDs = 0,1,2
    entity_types = np.concatenate(
        [
            np.zeros((200,)),
            np.ones((60,)),
            2 * np.ones((n_entity - 260)),
        ]
    ).astype(np.int32)
    type_offsets = np.array([0, 200, 260])
    sharding = Sharding.create(n_entity, n_shard, seed=seed, type_offsets=type_offsets)

    # Types of head and tail entitites
    triple_types = np.random.randint(n_type, size=(n_triple, 2)).astype(np.int32)

    ns = TypeBasedShardedNegativeSampler(
        triple_types=triple_types,
        n_negative=n_negative,
        sharding=sharding,
        corruption_scheme=corruption_scheme,
        local_sampling=local_sampling,
        seed=seed,
    )

    sample_idx = np.random.randint(
        n_triple, size=sample_idx_size[triple_partition_mode]
    )
    neg_batch = ns(sample_idx)["negative_entities"]

    for processing_shard in range(n_shard):
        # Check sampled entities idx are smaller than
        # number of actual entitites in the shard
        assert (
            neg_batch[:, processing_shard].max()
            < sharding.shard_counts[processing_shard]
        )

        if ns.local_sampling:
            # All negative entities for a triple are sampled from processing_shard;
            # no all_to_all
            negative_types = entity_types[
                sharding.shard_and_idx_to_entity[
                    processing_shard, neg_batch[:, processing_shard, :, :, :]
                ]
            ]
        else:
            # Negative entities for a triple are sampledfrom all shards;
            # all_to_all (permuting neg_batch.shape[1] and neg_batch.shape[2])
            negative_types = entity_types[
                sharding.shard_and_idx_to_entity[
                    np.arange(n_shard)[None, :, None, None],
                    neg_batch[:, :, processing_shard, :, :],
                ]
            ]

        per_triple_unique_types = np.unique(negative_types, axis=-1)
        # Check that all negative entities sampled for a triple
        # on a specific shard have same type
        assert per_triple_unique_types.shape[-1] == 1
        negative_types = per_triple_unique_types.squeeze(-1)

        per_shard_unique_types = np.unique(negative_types, axis=1)
        # Check that negative entities sampled for a triple
        # on all shards have sampe type
        assert per_shard_unique_types.shape[1] == 1
        negative_types = per_shard_unique_types.squeeze(1)

        # Check that the negative types are correct
        if ns.corruption_scheme in ["h", "t"]:
            true_types = triple_types[
                einops.rearrange(
                    sample_idx[:, processing_shard],
                    "step ... triple -> step (... triple)",
                )
            ][..., 0 if ns.corruption_scheme == "h" else 1]
        elif ns.corruption_scheme == "ht":
            # Each shard-pair is split into two halves: corrupt heads in one,
            # tails in the other
            cutpoint = positive_per_partition // 2
            head_types = triple_types[sample_idx[:, processing_shard, ..., :cutpoint]][
                ..., 0
            ]
            tail_types = triple_types[sample_idx[:, processing_shard, ..., cutpoint:]][
                ..., 1
            ]
            true_types = einops.rearrange(
                np.concatenate([head_types, tail_types], axis=-1),
                "step ... triple -> step (... triple)",
            )
        assert_equal(negative_types, true_types)


@pytest.mark.parametrize("triple_partition_mode", ["shard", "shardpair"])
@pytest.mark.parametrize("corruption_scheme", ["h", "t", "ht"])
@pytest.mark.parametrize("flat_negative_format", [True, False])
def test_triple_based_sharded_ns(
    triple_partition_mode: str, corruption_scheme: str, flat_negative_format: bool
) -> None:
    np.random.seed(seed)

    sharding = Sharding.create(n_entity, n_shard, seed=seed)

    negative_heads = np.random.randint(
        n_entity, size=(1 if flat_negative_format else n_triple, n_negative)
    ).astype(np.int32)
    negative_tails = np.random.randint(
        n_entity, size=(1 if flat_negative_format else n_triple, n_negative)
    ).astype(np.int32)

    ns = TripleBasedShardedNegativeSampler(
        negative_heads,
        negative_tails,
        sharding,
        corruption_scheme=corruption_scheme,
        seed=seed,
        return_sort_idx=True,
        mask_on_gather=False,
    )

    sample_idx = np.random.randint(
        n_triple, size=sample_idx_size[triple_partition_mode]
    )
    neg_batch = ns(sample_idx)
    neg_batch_ent = neg_batch["negative_entities"].astype(np.int32)  # mypy check
    neg_batch_mask = neg_batch["negative_mask"].astype(np.bool_)
    neg_sort_idx = neg_batch["negative_sort_idx"].astype(np.int32)

    # Check that for all triples there are exactly n_negative entities in mask
    assert np.all(neg_batch_mask.sum(axis=(-2, -1)) == n_negative)

    if flat_negative_format:
        # Replicate entities and mask along microbatch dimension
        sample_idx = np.full_like(sample_idx, fill_value=0)
        shard_bs = np.prod(sample_idx.shape[2:])
        if corruption_scheme == "ht":
            neg_batch_h, neg_batch_t = np.split(neg_batch_ent, [1], axis=-2)
            neg_mask_h, neg_mask_t = np.split(neg_batch_mask, [1], axis=-3)
            if triple_partition_mode == "shardpair":
                neg_batch_h = np.expand_dims(neg_batch_h, 3).repeat(n_shard, axis=3)
                neg_batch_t = np.expand_dims(neg_batch_t, 3).repeat(n_shard, axis=3)
                neg_mask_h = np.expand_dims(neg_mask_h, 2).repeat(n_shard, axis=2)
                neg_mask_t = np.expand_dims(neg_mask_t, 2).repeat(n_shard, axis=2)
            neg_batch_ent = np.concatenate(
                [
                    np.repeat(neg_batch_h, cutpoint, axis=-2),
                    np.repeat(neg_batch_t, positive_per_partition - cutpoint, axis=-2),
                ],
                axis=-2,
            )
            neg_batch_mask = np.concatenate(
                [
                    np.repeat(neg_mask_h, cutpoint, axis=-3),
                    np.repeat(neg_mask_t, positive_per_partition - cutpoint, axis=-3),
                ],
                axis=-3,
            )
            neg_batch_ent = einops.rearrange(
                neg_batch_ent,
                "bps shard_neg shard ... triple neg ->"
                " bps shard_neg shard (... triple) neg",
            )
            neg_batch_mask = einops.rearrange(
                neg_batch_mask,
                "bps shard ... triple shard_neg neg ->"
                " bps shard (... triple) shard_neg neg",
            )
        else:
            neg_batch_ent = einops.repeat(
                neg_batch_ent,
                "... pad neg -> ... (r pad) neg",
                r=shard_bs,
            )
            neg_batch_mask = einops.repeat(
                neg_batch_mask,
                "... pad shard_neg neg -> ... (r pad) shard_neg neg",
                r=shard_bs,
            )

    for processing_shard in range(n_shard):
        # Check sampled entities idx are smaller than number of
        # actual entitites in the shard
        assert (
            neg_batch_ent[:, processing_shard].max()
            < sharding.shard_counts[processing_shard]
        )

        triple_on_shard_idx = einops.rearrange(
            sample_idx[:, processing_shard],
            "step ... triple -> step (... triple)",
        )

        # Negative entitites sent to processing_shard (after all_to_all)
        negative_entities = sharding.shard_and_idx_to_entity[
            np.arange(n_shard)[None, :, None, None],
            neg_batch_ent[:, :, processing_shard, :, :],
        ]
        negative_entities = einops.rearrange(
            negative_entities,
            "step shard_neg triple neg -> step triple (shard_neg neg)",
        )
        # Discard padding negatives
        mask = einops.rearrange(
            neg_batch_mask[:, processing_shard],
            "step triple shard_neg neg -> step triple (shard_neg neg)",
        )
        negative_entities_filtered = negative_entities[mask].reshape(
            *negative_entities.shape[:-1], -1
        )  # shape (step, shard_bs, n_negative)

        # Check that the set of triple-specific negatives is correct
        if corruption_scheme == "h":
            true_negatives = negative_heads[triple_on_shard_idx]
        elif corruption_scheme == "t":
            true_negatives = negative_tails[triple_on_shard_idx]
        elif corruption_scheme == "ht":
            # Each triple partition is split in two halves: return negative heads in one,
            # negative tails in the other
            head_negatives = negative_heads[
                sample_idx[:, processing_shard, ..., :cutpoint]
            ]
            tail_negatives = negative_tails[
                sample_idx[:, processing_shard, ..., cutpoint:]
            ]
            true_negatives = einops.rearrange(
                np.concatenate([head_negatives, tail_negatives], axis=-2),
                "step ... triple neg -> step (... triple) neg",
            )

        # Use neg_sort_idx to align the order of negatives for each triple
        assert_equal(
            negative_entities_filtered,
            np.take_along_axis(
                true_negatives, neg_sort_idx[:, processing_shard], axis=-1
            ),
        )
