# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
from numpy.testing import assert_equal

from besskge.dataset import KGDataset
from besskge.sharding import PartitionedTripleSet, Sharding

seed = 1234
n_entity = 501
n_relation_type = 11
n_shard = 5
n_triple = 2001
n_negative = 251

np.random.seed(seed)

type_offsets = np.array([0, 20, 60, 200, 341, 342])
triples_h = np.random.randint(n_entity, size=(n_triple,))
triples_t = np.random.randint(n_entity, size=(n_triple,))
triples_r = np.random.randint(n_relation_type, size=(n_triple,))
part = "test"
triples = {part: np.stack([triples_h, triples_r, triples_t], axis=1)}
neg_heads = {
    part: np.random.randint(n_entity, size=(n_triple, n_negative), dtype=np.int32)
}
neg_tails = {part: np.random.randint(n_entity, size=(1, n_negative), dtype=np.int32)}

ds = KGDataset(
    n_entity=n_entity,
    n_relation_type=n_relation_type,
    entity_dict=None,
    relation_dict=None,
    type_offsets={str(i): o for i, o in enumerate(type_offsets)},
    triples=triples,
    neg_heads=neg_heads,
    neg_tails=neg_tails,
)


def test_entity_sharding() -> None:
    np.random.seed(seed)

    sharding = Sharding.create(
        n_entity,
        n_shard,
        seed=seed,
        type_offsets=type_offsets,
    )

    assert_equal(sharding.n_shard, n_shard)
    assert_equal(sharding.n_entity, n_entity)
    assert_equal(sharding.max_entity_per_shard, np.ceil(n_entity / n_shard))
    assert_equal(np.unique(sharding.entity_to_shard), np.arange(n_shard))
    assert_equal(
        np.unique(sharding.entity_to_idx), np.arange(sharding.max_entity_per_shard)
    )

    # Shard counts
    assert_equal(sharding.shard_counts.sum(), n_entity)
    if sharding.entity_type_counts is not None:
        assert_equal(sharding.entity_type_counts.sum(-1), sharding.shard_counts)

    # Revert sharding
    assert_equal(
        sharding.shard_and_idx_to_entity[
            sharding.entity_to_shard, sharding.entity_to_idx
        ],
        np.arange(n_entity),
    )


@pytest.mark.parametrize("add_inverse_triples", [True, False])
def test_triple_partitioning_from_dataset(add_inverse_triples: bool) -> None:
    np.random.seed(seed)

    sharding = Sharding.create(
        n_entity,
        n_shard,
        seed=seed,
        type_offsets=type_offsets,
    )

    h_triple_set = PartitionedTripleSet.create_from_dataset(
        ds, part, sharding, "h_shard", add_inverse_triples=add_inverse_triples
    )
    t_triple_set = PartitionedTripleSet.create_from_dataset(
        ds, part, sharding, "t_shard", add_inverse_triples=add_inverse_triples
    )
    ht_triple_set = PartitionedTripleSet.create_from_dataset(
        ds, part, sharding, "ht_shardpair", add_inverse_triples=add_inverse_triples
    )

    for triple_set, partition_mode in zip(
        [h_triple_set, t_triple_set, ht_triple_set],
        ["h_shard", "t_shard", "ht_shardpair"],
    ):
        assert triple_set.dummy == "none"
        n_triple_tot = 2 * n_triple if add_inverse_triples else n_triple
        assert_equal(
            triple_set.triple_counts.sum(),
            n_triple_tot,
        )
        trip = triples[part]
        if add_inverse_triples:
            inv_triples = np.copy(trip[:, ::-1])
            inv_triples[:, 1] += n_relation_type
            trip = np.concatenate([trip, inv_triples], axis=0)
        # Put original triples in the same order as triples_set.triples
        triple_sorted = trip[triple_set.triple_sort_idx]
        # Check triple partitioning respects h/t sharding
        for h_shard in range(n_shard):
            for t_shard in range(n_shard):
                if partition_mode == "h_shard":
                    triple_range_l = triple_set.triple_offsets[h_shard]
                    triple_range_u = (
                        n_triple_tot
                        if h_shard == n_shard - 1
                        else triple_set.triple_offsets[h_shard + 1]
                    )
                    assert np.all(
                        sharding.entity_to_shard[
                            triple_sorted[triple_range_l:triple_range_u, 0]
                        ]
                        == h_shard
                    )
                elif partition_mode == "t_shard":
                    triple_range_l = triple_set.triple_offsets[t_shard]
                    triple_range_u = (
                        n_triple_tot
                        if t_shard == n_shard - 1
                        else triple_set.triple_offsets[t_shard + 1]
                    )
                    assert np.all(
                        sharding.entity_to_shard[
                            triple_sorted[triple_range_l:triple_range_u, 2]
                        ]
                        == t_shard
                    )
                else:
                    triple_range_l = triple_set.triple_offsets.flatten()[
                        n_shard * h_shard + t_shard
                    ]
                    triple_range_u = (
                        n_triple_tot
                        if h_shard == n_shard - 1 and t_shard == n_shard - 1
                        else triple_set.triple_offsets.flatten()[
                            n_shard * h_shard + t_shard + 1
                        ]
                    )
                    assert np.all(
                        sharding.entity_to_shard[
                            triple_sorted[triple_range_l:triple_range_u, 0]
                        ]
                        == h_shard
                    )
                    assert np.all(
                        sharding.entity_to_shard[
                            triple_sorted[triple_range_l:triple_range_u, 2]
                        ]
                        == t_shard
                    )

        if ds.ht_types:
            types = ds.ht_types[part]
            if add_inverse_triples:
                types = np.concatenate([types, types[:, ::-1]], axis=0)
            types_sorted = types[triple_set.triple_sort_idx]
            assert_equal(types_sorted, triple_set.types)
        negative_heads = neg_heads[part]
        if add_inverse_triples:
            negative_heads = np.concatenate(
                [
                    negative_heads,
                    np.broadcast_to(neg_tails[part], negative_heads.shape),
                ],
                axis=0,
            )
            neg_tails_sorted = np.concatenate(
                [
                    np.broadcast_to(neg_tails[part], neg_heads[part].shape),
                    neg_heads[part],
                ],
                axis=0,
            )[triple_set.triple_sort_idx]
            assert_equal(neg_tails_sorted, triple_set.neg_tails)
        else:
            assert_equal(neg_tails[part], triple_set.neg_tails)
        neg_heads_sorted = negative_heads[triple_set.triple_sort_idx]
        assert_equal(neg_heads_sorted, triple_set.neg_heads)
        # Pass from global IDs to local IDs
        if partition_mode in ["h_shard", "ht_shardpair"]:
            triple_sorted[:, 0] = sharding.entity_to_idx[triple_sorted[:, 0]]
        if partition_mode in ["t_shard", "ht_shardpair"]:
            triple_sorted[:, 2] = sharding.entity_to_idx[triple_sorted[:, 2]]
        assert_equal(triple_sorted, triple_set.triples)

    if add_inverse_triples:
        h_triple_set_noinv = PartitionedTripleSet.create_from_dataset(
            ds, part, sharding, "h_shard", add_inverse_triples=False
        )
        t_triple_set_noinv = PartitionedTripleSet.create_from_dataset(
            ds, part, sharding, "t_shard", add_inverse_triples=False
        )
        ht_triple_set_noinv = PartitionedTripleSet.create_from_dataset(
            ds, part, sharding, "ht_shardpair", add_inverse_triples=False
        )

        assert_equal(
            ht_triple_set.triple_counts,
            ht_triple_set_noinv.triple_counts + ht_triple_set_noinv.triple_counts.T,
        )
        assert_equal(
            h_triple_set.triple_counts,
            h_triple_set_noinv.triple_counts + t_triple_set_noinv.triple_counts,
        )
        assert_equal(
            t_triple_set.triple_counts,
            h_triple_set_noinv.triple_counts + t_triple_set_noinv.triple_counts,
        )


@pytest.mark.parametrize("query_mode", ["hr", "rt"])
@pytest.mark.parametrize("negative_type", [True, False])
def test_triple_partitioning_from_queries(query_mode: str, negative_type: bool) -> None:
    np.random.seed(seed)

    sharding = Sharding.create(
        n_entity,
        n_shard,
        seed=seed,
        type_offsets=type_offsets,
    )

    if query_mode == "hr":
        queries = triples[part][:, :2]
        ground_truth = triples[part][:, 2]
        if negative_type:
            trip = np.concatenate(
                [
                    queries,
                    np.full(fill_value=type_offsets[1], shape=(queries.shape[0], 1)),
                ],
                axis=1,
            )
        else:
            trip = np.concatenate([queries, ground_truth.reshape(-1, 1)], axis=1)
        negatives = neg_tails[part]
    elif query_mode == "rt":
        queries = triples[part][:, 1:]
        ground_truth = triples[part][:, 0]
        if negative_type:
            trip = np.concatenate(
                [
                    np.full(fill_value=type_offsets[1], shape=(queries.shape[0], 1)),
                    queries,
                ],
                axis=1,
            )
        else:
            trip = np.concatenate([ground_truth.reshape(-1, 1), queries], axis=1)
        negatives = neg_heads[part]

    triple_set = PartitionedTripleSet.create_from_queries(
        ds,
        sharding,
        queries,
        query_mode,
        ground_truth if not negative_type else None,
        negatives,
        negative_type="1" if negative_type else None,
    )

    triple_sorted = trip[triple_set.triple_sort_idx]
    if query_mode == "hr":
        triple_sorted[:, 0] = sharding.entity_to_idx[triple_sorted[:, 0]]
        assert_equal(triple_set.neg_tails, negatives)
        assert triple_set.neg_heads is None
    elif query_mode == "rt":
        triple_sorted[:, 2] = sharding.entity_to_idx[triple_sorted[:, 2]]
        assert_equal(triple_set.neg_heads, negatives[triple_set.triple_sort_idx])
        assert triple_set.neg_tails is None
    assert_equal(triple_sorted, triple_set.triples)
