# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import dataclasses
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import ogb.linkproppred


@dataclasses.dataclass
class KGDataset:
    """
    Represents a complete knowledge graph dataset of (head, relation, tail) triples.
    """

    #: Number of entities (nodes) in the KG
    n_entity: int

    #: Number of relation types (edge labels) in the KG
    n_relation_type: int

    #: List of (h_ID, r_ID, t_ID) triples, for each part of the dataset;
    # {part: int64[n_triple, {h,r,t}]}
    triples: Dict[str, np.ndarray]

    #: Entity labels by ID; str[n_entity]
    entity_dict: Optional[np.ndarray]

    #: Relation type labels by ID; str[n_relation_type]
    relation_dict: Optional[np.ndarray]

    #: If entities have types, IDs are assumed to be clustered by type;
    # {entity_type: int}
    type_offsets: Optional[Dict[str, int]]

    #: Type IDs of head/tail entities;
    # {part: int64[n_triple, {h_type,t_type}]}
    types: Optional[Dict[str, np.ndarray]]

    #: IDs of triple-specific negative heads;
    # {part: int64[n_triple, n_neg_heads]}
    neg_heads: Optional[Dict[str, np.ndarray]]

    #: IDs of triple-specific negative heads;
    # {part: int64[n_triple, n_neg_tails]}
    neg_tails: Optional[Dict[str, np.ndarray]]

    @classmethod
    def build_biokg(cls, root: Path) -> "KGDataset":
        """
        Build the OGB-BioKG dataset.

        :param root:
            Path to dataset.

        :return: OGB-BioKG KGDataset.
        """
        dataset = ogb.linkproppred.LinkPropPredDataset(name="ogbl-biokg", root=root)
        split_edge = dataset.get_edge_split()
        n_relation_type = len(dataset[0]["edge_reltype"].keys())
        type_counts = dataset[0]["num_nodes_dict"]
        type_offsets = np.concatenate(
            ([0], np.cumsum(np.fromiter(type_counts.values(), dtype=int)))
        )
        n_entity = type_offsets[-1]
        type_offsets = dict(zip(type_counts.keys(), type_offsets))
        triples = {}
        types = {}
        neg_heads = {}
        neg_tails = {}
        for part, hrt in split_edge.items():
            h_label, h_type_idx = np.unique(hrt["head_type"], return_inverse=True)
            t_label, t_type_idx = np.unique(hrt["tail_type"], return_inverse=True)
            types[part] = np.stack([h_type_idx, t_type_idx], axis=-1)
            h_type_offsets = np.array([type_offsets[lab] for lab in h_label])
            t_type_offsets = np.array([type_offsets[lab] for lab in t_label])
            hrt["head"] += h_type_offsets[h_type_idx]
            hrt["tail"] += t_type_offsets[t_type_idx]
            triples[part] = np.stack(
                [hrt["head"], hrt["relation"], hrt["tail"]], axis=-1
            )
            if part != "train":
                neg_heads[part] = hrt["head_neg"] + h_type_offsets[h_type_idx][:, None]
                neg_tails[part] = hrt["tail_neg"] + t_type_offsets[t_type_idx][:, None]

        return cls(
            n_entity=n_entity,
            n_relation_type=n_relation_type,
            entity_dict=None,
            relation_dict=None,
            type_offsets=type_offsets,
            triples=triples,
            types=types,
            neg_heads=neg_heads,
            neg_tails=neg_tails,
        )

    @classmethod
    def load(cls, path: Path) -> "KGDataset":
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


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
