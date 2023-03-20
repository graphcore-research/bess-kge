import dataclasses
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from besskge.dataset import KGDataset


@dataclasses.dataclass
class Sharding:
    """
    A mapping of entities to shards (and back again).
    """

    #: Number of shards
    n_shard: int

    #: Entity shard by global ID; int64[n_entity]
    entity_to_shard: np.ndarray

    #: Entity local ID on shard by global ID; int64[n_entity]
    entity_to_idx: np.ndarray

    #: Entity global ID by (shard, local_ID); int64[n_shard, max_entity_per_shard]
    shard_and_idx_to_entity: np.ndarray

    #: Number of true entities (excluding padding) in each shard; int64[n_shard]
    shard_counts: np.ndarray

    #: Number of entities of each type on each shard; int64[n_shard, n_types]
    entity_type_counts: Optional[np.ndarray]

    #: Entities of same type remain clustered also locally; int64[n_shard, n_types]
    entity_type_offsets: Optional[np.ndarray]

    @property
    def n_entity(self) -> int:
        return len(self.entity_to_shard)

    @property
    def max_entity_per_shard(self) -> int:
        return self.shard_and_idx_to_entity.shape[1]

    @classmethod
    def create(
        cls,
        n_entity: int,
        n_shard: int,
        seed: int,
        type_offsets: Optional[np.ndarray] = None,
    ) -> "Sharding":
        """
        Construct a random balanced sharding of entities.

        :param n_entity:
            Number of entities in the KG.
        :param n_shard:
            Number of shards.
        :param seed:
            Seed for random sharding.
        :param type_offsets: shape: (n_types,)
            Global offsets of entity types, defaults to None.

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

    def save(self, path: Path) -> None:
        """
        Save sharding to .npz file.

        :param path:
            Save path.
        """
        np.savez(path, **dataclasses.asdict(self))

    @classmethod
    def load(cls, path: Path) -> "Sharding":
        """
        Load a :class:`Sharding` saved with :func:`Sharding.save`.

        :param path:
            Path to saved :class:`Sharding`.

        :return:
            The saved :class:`Sharding`.
        """
        data = dict(np.load(path))
        return cls(n_shard=int(data.pop("n_shard")), **data)


@dataclasses.dataclass
class ShardedTripleSet:
    """
    A collection of sharded triples.
    Each triple (h,r,t) is assigned to one of n_shard ** 2 shardpairs
    based on the entity sharding of h and t: (shard_h, shard_t).
    Shardpairs are ordered as:
    (0,0), (0,1), ..., (0,n_shard-1), (1,0), ..., (n_shard-1, n_shard-1).
    """

    #: Sharding of entities
    sharding: Sharding

    #: Triples ordered by shardpair (local IDs for entities)
    # int64[n_triple, {h,r,t}]
    sharded_triples: np.ndarray

    #: Number of triples per shardpair
    # int64[n_shard, n_shard]
    shardpair_counts: np.ndarray

    #: Shardpair delimiting indices
    # int64[n_shard, n_shard]
    shardpair_offsets: np.ndarray

    #: Sorting indices used to order triples by shardpair
    # int64[n_triple]
    sort_idx: np.ndarray

    #: Entity type IDs of triple head/tail
    # int64[n_triple, {h_type, t_type}]
    types: Optional[np.ndarray]

    #: Global IDs of triple-specific negative heads;
    # int64[n_triple, n_neg_heads]
    neg_heads: Optional[np.ndarray]

    #: Global IDs of triple-specific negative heads;
    # int64[n_triple, n_neg_tails]
    neg_tails: Optional[np.ndarray]

    @classmethod
    def create(
        cls,
        dataset: KGDataset,
        part: str,
        sharding: Sharding,
    ) -> "ShardedTripleSet":
        """
        Shard triples wrt an entity sharding.

        :param dataset:
            KG dataset.
        :param part:
            The dataset part to shard.
        :param sharding:
            The entity sharding to use.

        :return:
            Sharded set of triples.
        """

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

        if dataset.types and part in dataset.types.keys():
            types = dataset.types[part][sort_idx]
        else:
            types = None

        if dataset.neg_heads and part in dataset.neg_heads.keys():
            neg_heads = dataset.neg_heads[part][sort_idx]
        else:
            neg_heads = None

        if dataset.neg_tails and part in dataset.neg_tails.keys():
            neg_tails = dataset.neg_tails[part][sort_idx]
        else:
            neg_tails = None

        return cls(
            sharding=sharding,
            sharded_triples=sharded_triples,
            shardpair_counts=shardpair_counts,
            shardpair_offsets=shardpair_offsets,
            sort_idx=sort_idx,
            types=types,
            neg_heads=neg_heads,
            neg_tails=neg_tails,
        )
