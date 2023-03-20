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
positive_per_shardpair = 60
n_negative = 250


@pytest.mark.parametrize("flat_negative_format", [True, False])
def test_random_sharded_ns(flat_negative_format: bool) -> None:
    np.random.seed(seed)
    sharding = Sharding.create(n_entity, n_shard, seed=seed)

    ns = RandomShardedNegativeSampler(
        n_negative=n_negative,
        sharding=sharding,
        seed=seed,
        corruption_scheme="h",
        local_sampling=False,
        flat_negative_format=flat_negative_format,
    )

    # Negative batch is independent on sample_idx, corruption_scheme and local_sampling
    sample_idx = np.ones((batches_per_step, n_shard, n_shard, positive_per_shardpair))
    neg_batch = ns.get_negative_batch(sample_idx)["negative_entities"]

    # Check sampled entities idx are smaller than number of actual entitites in the shard
    for processing_shard in range(n_shard):
        assert (
            neg_batch[:, processing_shard].max()
            < sharding.shard_counts[processing_shard]
        )


@pytest.mark.parametrize("local_sampling", [True, False])
@pytest.mark.parametrize("corruption_scheme", ["h", "t", "ht"])
def test_type_based_sharded_ns(local_sampling: bool, corruption_scheme: str) -> None:
    np.random.seed(seed)
    n_type = 3
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
    triple_types = np.random.randint(n_type, size=(n_triple, 2))

    ns = TypeBasedShardedNegativeSampler(
        triple_types=triple_types,
        n_negative=n_negative,
        sharding=sharding,
        corruption_scheme=corruption_scheme,
        local_sampling=local_sampling,
        seed=seed,
    )

    sample_idx = np.random.randint(
        n_triple, size=(batches_per_step, n_shard, n_shard, positive_per_shardpair)
    )
    neg_batch = ns.get_negative_batch(sample_idx)["negative_entities"]

    for processing_shard in range(n_shard):
        # Check sampled entities idx are smaller than number of actual entitites in the shard
        assert (
            neg_batch[:, processing_shard].max()
            < sharding.shard_counts[processing_shard]
        )

        if ns.local_sampling:
            # All negative entities for a triple are sampled from processing_shard; no all_to_all
            negative_types = entity_types[
                sharding.shard_and_idx_to_entity[
                    processing_shard, neg_batch[:, processing_shard, :, :, :]
                ]
            ]
        else:
            # Negative entities for a triple are sampled from all shards; all_to_all (permuting neg_batch.shape[1] and neg_batch.shape[2])
            negative_types = entity_types[
                sharding.shard_and_idx_to_entity[
                    np.arange(n_shard)[None, :, None, None],
                    neg_batch[:, :, processing_shard, :, :],
                ]
            ]

        per_triple_unique_types = np.unique(negative_types, axis=-1)
        # Check that all negative entities sampled for a triple on a specific shard have same type
        assert per_triple_unique_types.shape[-1] == 1
        negative_types = per_triple_unique_types.squeeze(-1)

        per_shard_unique_types = np.unique(negative_types, axis=1)
        # Check that negative entities sampled for a triple on all shards have sampe type
        assert per_shard_unique_types.shape[1] == 1
        negative_types = per_shard_unique_types.squeeze(1)

        # Check that the negative types are correct
        if ns.corruption_scheme in ["h", "t"]:
            true_types = triple_types[
                einops.rearrange(
                    sample_idx[:, processing_shard],
                    "step tail_shard triple -> step (tail_shard triple)",
                )
            ][..., 0 if ns.corruption_scheme == "h" else 1]
        elif ns.corruption_scheme == "ht":
            # Each shardpair is split into two halves: corrupt heads in one, tails in the other
            cutpoint = positive_per_shardpair // 2
            head_types = triple_types[sample_idx[:, processing_shard, :, :cutpoint]][
                ..., 0
            ]
            tail_types = triple_types[sample_idx[:, processing_shard, :, cutpoint:]][
                ..., 1
            ]
            true_types = einops.rearrange(
                np.concatenate([head_types, tail_types], axis=2),
                "step tail_shard triple -> step (tail_shard triple)",
            )
        assert_equal(negative_types, true_types)


@pytest.mark.parametrize("corruption_scheme", ["h", "t", "ht"])
def test_triple_based_sharded_ns(corruption_scheme: str) -> None:
    np.random.seed(seed)

    sharding = Sharding.create(n_entity, n_shard, seed=seed)

    negative_heads = np.random.randint(n_entity, size=(n_triple, n_negative))
    negative_tails = np.random.randint(n_entity, size=(n_triple, n_negative))

    ns = TripleBasedShardedNegativeSampler(
        negative_heads,
        negative_tails,
        sharding,
        corruption_scheme=corruption_scheme,
        seed=seed,
        get_sort_idx=True,
    )

    sample_idx = np.random.randint(
        n_triple, size=(batches_per_step, n_shard, n_shard, positive_per_shardpair)
    )
    neg_batch = ns.get_negative_batch(sample_idx)
    neg_batch_ent = neg_batch["negative_entities"]
    neg_batch_mask = neg_batch["negative_mask"]
    neg_sort_idx = neg_batch["negative_sort_idx"]

    # Check that for all triples there are exactly n_negative True mask entries
    assert_equal(neg_batch_mask.sum(-1), n_negative)

    for processing_shard in range(n_shard):
        # Check sampled entities idx are smaller than number of actual entitites in the shard
        assert (
            neg_batch_ent[:, processing_shard].max()
            < sharding.shard_counts[processing_shard]
        )

        triple_on_shard_idx = einops.rearrange(
            sample_idx[:, processing_shard],
            "step tail_shard triple -> step (tail_shard triple)",
        )

        # Negative entitites sent to processing_shard (after all_to_all)
        negative_entities = sharding.shard_and_idx_to_entity[
            np.arange(n_shard)[None, :, None, None],
            neg_batch_ent[:, :, processing_shard, :, :],
        ]
        negative_entities = einops.rearrange(
            negative_entities,
            "step source_shard triple neg -> step triple (source_shard neg)",
        )
        # Discard padding negatives
        negative_entities_filtered = einops.rearrange(
            negative_entities[neg_batch_mask[:, processing_shard]],
            "(step triple neg) -> step triple neg",
            step=batches_per_step,
            neg=n_negative,
        )

        # Check that the set of triple-specific negatives is correct
        if corruption_scheme == "h":
            true_negatives = negative_heads[triple_on_shard_idx]
        elif corruption_scheme == "t":
            true_negatives = negative_tails[triple_on_shard_idx]
        elif corruption_scheme == "ht":
            # Each shardpair is split into two halves: return negative heads in one, negative tails in the other
            cutpoint = positive_per_shardpair // 2
            head_negatives = negative_heads[
                sample_idx[:, processing_shard, :, :cutpoint]
            ]
            tail_negatives = negative_tails[
                sample_idx[:, processing_shard, :, cutpoint:]
            ]
            true_negatives = einops.rearrange(
                np.concatenate([head_negatives, tail_negatives], axis=2),
                "step tail_shard triple neg -> step (tail_shard triple) neg",
            )
        # Use neg_sort_idx to align the order of negatives for each triple
        assert_equal(
            negative_entities_filtered,
            np.take_along_axis(
                true_negatives, neg_sort_idx[:, processing_shard], axis=-1
            ),
        )
