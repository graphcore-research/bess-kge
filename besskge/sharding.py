# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import dataclasses
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from besskge.dataset import KGDataset


@dataclasses.dataclass
class Sharding:
    """
    A mapping of entities to shards (and back again).
    """

    #: Number of shards
    n_shard: int

    #: Entity shard by global ID;
    #: int32[n_entity]
    entity_to_shard: NDArray[np.int32]

    #: Entity local ID on shard by global ID;
    #: int32[n_entity]
    entity_to_idx: NDArray[np.int32]

    #: Entity global ID by (shard, local_ID);
    #: int32[n_shard, max_entity_per_shard]
    shard_and_idx_to_entity: NDArray[np.int32]

    #: Number of true entities (excluding padding) in each shard;
    #: int64[n_shard]
    shard_counts: NDArray[np.int64]

    #: Number of entities of each type on each shard;
    #: int64[n_shard, n_types]
    entity_type_counts: Optional[NDArray[np.int64]]

    #: Offsets for entities of same type on each shared
    #: (entities remain clustered by type also locally);
    #: int64[n_shard, n_types]
    entity_type_offsets: Optional[NDArray[np.int64]]

    @property
    def n_entity(self) -> int:
        """
        Number of entities in the knowledge graph.
        """
        return len(self.entity_to_shard)

    @property
    def max_entity_per_shard(self) -> int:
        """
        Number of entities (including possibly one final padding) in each shard.
        """
        return self.shard_and_idx_to_entity.shape[1]

    @classmethod
    def create(
        cls,
        n_entity: int,
        n_shard: int,
        seed: int,
        type_offsets: Optional[NDArray[np.int64]] = None,
    ) -> "Sharding":
        """
        Construct a random, balanced sharding of entities.

        :param n_entity:
            Number of entities in the knowledge graph.
        :param n_shard:
            Number of shards.
        :param seed:
            Seed for random sharding.
        :param type_offsets: shape: (n_types,)
            Global offsets of entity types. Default: None.

        :return:
            Random sharding of n_entity entities in n_shard shards.
        """
        rng = np.random.RandomState(seed)
        max_entity_per_shard = int(np.ceil(n_entity / n_shard))
        # Keep global entity ID sorted on each shard, to preserve type-based clustering
        shard_and_idx_to_entity = np.sort(
            rng.permutation(n_shard * max_entity_per_shard).reshape(
                n_shard, max_entity_per_shard
            ),
            axis=1,
        )
        entity_to_shard, entity_to_idx = np.divmod(
            np.argsort(shard_and_idx_to_entity.flatten())[:n_entity],
            max_entity_per_shard,
        )
        # Check whether the last entity in each shard is padding or not
        shard_deduction = shard_and_idx_to_entity[:, -1] >= n_entity
        # Number of actual entities in each shard
        shard_counts = max_entity_per_shard - shard_deduction

        entity_type_counts: Optional[NDArray[np.int64]]
        entity_type_offsets: Optional[NDArray[np.int64]]
        if type_offsets is not None:
            type_id_per_shard = (
                np.digitize(shard_and_idx_to_entity, bins=type_offsets)
                + len(type_offsets) * np.arange(n_shard)[:, None]
                - 1
            )
            # Per-shard entity type counts and offsets
            entity_type_counts = np.bincount(
                type_id_per_shard.flatten(), minlength=len(type_offsets) * n_shard
            ).reshape(n_shard, -1)
            entity_type_offsets = np.c_[
                [0] * n_shard, np.cumsum(entity_type_counts, axis=1)[:, :-1]
            ]
            entity_type_counts[:, -1] -= shard_deduction
        else:
            entity_type_counts = entity_type_offsets = None

        return cls(
            n_shard=n_shard,
            entity_to_shard=entity_to_shard,
            entity_to_idx=entity_to_idx,
            shard_and_idx_to_entity=shard_and_idx_to_entity,
            shard_counts=shard_counts,
            entity_type_counts=entity_type_counts,
            entity_type_offsets=entity_type_offsets,
        )

    def save(self, out_file: Path) -> None:
        """
        Save sharding to .npz file.

        :param out_file:
            Path to output file.
        """
        np.savez(out_file, **dataclasses.asdict(self))

    @classmethod
    def load(cls, path: Path) -> "Sharding":
        """
        Load a :class:`Sharding` object saved with :func:`Sharding.save`.

        :param path:
            Path to saved :class:`Sharding` object.

        :return:
            The saved :class:`Sharding` object.
        """
        data = dict(np.load(path))
        return cls(n_shard=int(data.pop("n_shard")), **data)


@dataclasses.dataclass
class PartitionedTripleSet:
    """
    A partitioned collection of triples.
    If :code:`partition_mode = 'h_shard'` each triple is assigned to one of
    `n_shard` partitions based on the shard where the head entity is stored.
    Similarly, if :code:`partition_mode = 't_shard'`, each triple is assigned
    to one of `n_shard` partitions based on the shard where the tail entity is
    stored.

    If :code:`partition_mode = 'ht_shardpair'`, each triple is assigned to one
    of `n_shard^2` partitions based on the shard-pair `(shard_h, shard_t)`.
    Shard-pairs are ordered as:
    `(0,0), (0,1), ..., (0, n_shard-1), (1,0), ..., (n_shard-1, n_shard-1)`.
    """

    #: Sharding of entities
    sharding: Sharding

    #: Partitioning criterion for triples;
    #: "h_shard", "t_shard", "ht_shardpair"
    partition_mode: str

    #: If set is constructed from (h,r,?) (resp. (?,r,t)) queries,
    #: dummy tails (resp. heads) are added to make pairs into triples.
    #: "head", "tail", "none"
    dummy: Optional[str]

    #: h/r/t IDs for triples ordered by partition.
    #: Local IDs for heads (resp. tails) and global IDs
    #: for tails (resp. heads) if partition_mode = "h_shard" (resp. "t_shard");
    #: local IDs for heads and tails if partition_mode = "ht_shardpair"
    #: int32[n_triple, {h,r,t}]
    triples: NDArray[np.int32]

    #: Number of triples in each partition;
    #: int64[n_shard] or int64[n_shard, n_shard]
    triple_counts: NDArray[np.int64]

    #: Delimiting indices of ordered partitions;
    #: int64[n_shard] or int64[n_shard, n_shard]
    triple_offsets: NDArray[np.int64]

    #: Sorting indices to order triples by partition;
    #: int64[n_triple]
    triple_sort_idx: NDArray[np.int64]

    #: Entity type IDs of triple head/tail;
    #: int32[n_triple, {h_type, t_type}]
    types: Optional[NDArray[np.int32]]

    #: Global IDs of (possibly triple-specific) negative heads;
    #: int32[n_triple or 1, n_neg_heads]
    neg_heads: Optional[NDArray[np.int32]]

    #: Global IDs of (possibly triple-specific) negative heads;
    #: int32[n_triple or 1, n_neg_tails]
    neg_tails: Optional[NDArray[np.int32]]

    @classmethod
    def partition_triples(
        cls, triples: NDArray[np.int32], sharding: Sharding, partition_mode: str
    ) -> Tuple[
        NDArray[np.int32], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]
    ]:
        n_shard = sharding.n_shard
        offsets: NDArray[np.int64]
        if partition_mode in ["h_shard", "t_shard"]:
            column_id = 0 if partition_mode == "h_shard" else -1
            partition_id = sharding.entity_to_shard[triples[:, column_id]]
            counts = np.bincount(partition_id, minlength=n_shard)
            offsets = np.concatenate([np.array([0]), np.cumsum(counts)[:-1]])
        elif partition_mode == "ht_shardpair":
            shard_h, shard_t = sharding.entity_to_shard[triples[:, [0, 2]].T]
            partition_id = shard_h * n_shard + shard_t
            counts = np.bincount(partition_id, minlength=n_shard * n_shard).reshape(
                n_shard, n_shard
            )
            offsets = np.concatenate([np.array([0]), np.cumsum(counts)[:-1]]).reshape(
                n_shard, n_shard
            )
        else:
            raise ValueError(
                f"Partition mode {partition_mode} not supported"
                " for triple partitioning"
            )

        sort_idx = np.argsort(partition_id)
        sorted_triples: NDArray[np.int32]
        sorted_triples = triples[sort_idx]
        if partition_mode in ["h_shard", "ht_shardpair"]:
            sorted_triples[:, 0] = sharding.entity_to_idx[sorted_triples[:, 0]]
        if partition_mode in ["t_shard", "ht_shardpair"]:
            sorted_triples[:, -1] = sharding.entity_to_idx[sorted_triples[:, -1]]

        return sorted_triples, counts, offsets, sort_idx

    @classmethod
    def create_from_dataset(
        cls,
        dataset: KGDataset,
        part: str,
        sharding: Sharding,
        partition_mode: str = "ht_shardpair",
    ) -> "PartitionedTripleSet":
        """
        Create a partitioned triple set from a :class:`KGDataset` part.

        :param dataset:
            Knowledge graph dataset.
        :param part:
            The dataset part to shard.
        :param sharding:
            The entity sharding to use.
        :param partition_mode:
            The triple partition mode. Can be
            "h_shard", "t_shard", "ht_shardpair".

        :return:
            Partitioned set of triples.
        """

        triples = dataset.triples[part]

        (
            sorted_triples,
            counts,
            offsets,
            sort_idx,
        ) = PartitionedTripleSet.partition_triples(triples, sharding, partition_mode)

        ht_types = dataset.ht_types
        if ht_types and part in ht_types.keys():
            types = ht_types[part][sort_idx]
        else:
            types = None

        if dataset.neg_heads and part in dataset.neg_heads.keys():
            neg_heads = dataset.neg_heads[part]
            neg_heads = neg_heads.reshape(-1, neg_heads.shape[-1])
            if neg_heads.shape[0] != 1:
                neg_heads = neg_heads[sort_idx]
        else:
            neg_heads = None

        if dataset.neg_tails and part in dataset.neg_tails.keys():
            neg_tails = dataset.neg_tails[part]
            neg_tails = neg_tails.reshape(-1, neg_tails.shape[-1])
            if neg_tails.shape[0] != 1:
                neg_tails = neg_tails[sort_idx]
        else:
            neg_tails = None

        return cls(
            sharding=sharding,
            partition_mode=partition_mode,
            dummy="none",
            triples=sorted_triples,
            triple_counts=counts,
            triple_offsets=offsets,
            triple_sort_idx=sort_idx,
            types=types,
            neg_heads=neg_heads,
            neg_tails=neg_tails,
        )

    @classmethod
    def create_from_queries(
        cls,
        dataset: KGDataset,
        sharding: Sharding,
        queries: NDArray[np.int32],
        query_mode: str,
        ground_truth: Optional[NDArray[np.int32]] = None,
        negative: Optional[NDArray[np.int32]] = None,
        negative_type: Optional[str] = None,
    ) -> "PartitionedTripleSet":
        """
        Create a partitioned triple set from a set of (h,r,?) or (?,r,t) queries.
        Pairs are completed to triples by adding dummy entities.

        :param dataset:
            Knowledge graph dataset.
        :param sharding:
            The entity sharding to use.
        :param queries: shape: (n_query, 2)
            The set of (h, r) or (r, t) queries.
            Global IDs for entities/relations.
        :param query_mode:
            "hr" for (h,r,?) queries, "rt" for (?,r,t) queries.
        :param ground_truth: shape: (n_query,)
            If known, the global ID of the ground truth tail/head.
        :param negative: shape: (N, n_negative)
            Global IDs of negative entities to score against each query,
            query-specific (N=n_query) or the same for all queries (N=1).
            Default: None (namely the score queries against all entities in the graph).
        :param negative_type:
            Score queries only against entities of a specific type.
            Default: None (namely the score queries against entities of any type).

        :return:
            Partitioned set of queries (with dummy h/t completion).
        """

        n_query = queries.shape[0]
        # Dummy entities to complete queries (=pairs) to triples
        if negative_type:
            if (
                not dataset.type_offsets
                or negative_type not in dataset.type_offsets.keys()
            ):
                raise ValueError(
                    f"{negative_type} is not the label of"
                    " a type of entity in the KGDataset"
                )
            ds_type_offsets = dataset.type_offsets
            type_range_dict = {
                k: (a, b - 1)
                for ((k, a), b) in zip(
                    dataset.type_offsets.items(),
                    [*list(dataset.type_offsets.values())[1:], dataset.n_entity],
                )
            }
            type_range = type_range_dict[negative_type]
            if negative is not None:
                # Check that all negatives provided are of requested type
                if np.any(negative < type_range[0]) or np.any(
                    negative >= type_range[1]
                ):
                    warnings.warn(
                        "The negative entities provided are not all"
                        " of the specified negative_type"
                    )

        if ground_truth is not None:
            fill_column = ground_truth.reshape(n_query, 1)
        elif negative_type:
            fill_column = np.full(fill_value=type_range[0], shape=(n_query, 1))
        else:
            fill_column = np.full(fill_value=0, shape=(n_query, 1))

        if negative is not None:
            negative = negative.reshape(-1, negative.shape[-1])
        elif negative_type:
            negative = np.expand_dims(np.arange(type_range[0], type_range[1]), axis=0)
        else:
            negative = np.expand_dims(np.arange(sharding.n_entity), axis=0)

        if query_mode == "hr":
            triples = np.concatenate([queries, fill_column], axis=-1)
            partition_mode = "h_shard"
            dummy = "tail" if ground_truth is None else None
            neg_heads = None
            neg_tails = negative
        elif query_mode == "rt":
            triples = np.concatenate([fill_column, queries], axis=-1)
            partition_mode = "t_shard"
            dummy = "head" if ground_truth is None else None
            neg_heads = negative
            neg_tails = None
        else:
            raise ValueError(f"Query mode {query_mode} not supported")

        (
            sorted_triples,
            counts,
            offsets,
            sort_idx,
        ) = PartitionedTripleSet.partition_triples(triples, sharding, partition_mode)

        if negative_type:
            types = (
                np.digitize(
                    sorted_triples[:, [0, 2]],
                    np.fromiter(ds_type_offsets.values(), dtype=np.int32),
                )
                - 1
            )
        else:
            types = None

        if neg_heads is not None and neg_heads.shape[0] != 1:
            neg_heads = neg_heads[sort_idx]
        if neg_tails is not None and neg_tails.shape[0] != 1:
            neg_tails = neg_tails[sort_idx]

        return cls(
            sharding=sharding,
            partition_mode=partition_mode,
            dummy=dummy,
            triples=sorted_triples,
            triple_counts=counts,
            triple_offsets=offsets,
            triple_sort_idx=sort_idx,
            types=types,
            neg_heads=neg_heads,
            neg_tails=neg_tails,
        )
