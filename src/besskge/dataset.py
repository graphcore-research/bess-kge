from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import ogb.linkproppred


@dataclass
class KGDataset:
    """Represents a complete knowledge graph dataset of (head, relation, tail) triples."""

    n_entity: int
    n_relation_type: int
    entity_dict: np.ndarray  # str[n_entity]
    relation_dict: np.ndarray  # str[n_relation_type]
    type_offsets: Dict[str, int]  # {entity_type: int}
    triples: Dict[str, np.ndarray]  # {"train|valid|test": int64[n_triple, {h,r,t}]}
    types: Dict[
        str, np.ndarray
    ]  # {"train|valid|test": int64[n_triple, {h_type,t_type}]}
    neg_heads: Dict[
        str, np.ndarray
    ]  # {"train|valid|test": int64[n_triple, n_neg_heads]}
    neg_tails: Dict[
        str, np.ndarray
    ]  # {"train|valid|test": int64[n_triple, n_neg_tails]}

    @classmethod
    def build_biokg(cls, root: Path) -> "KGDataset":
        """Build the OGB-BioKG dataset."""
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

    # @classmethod
    # def build_wikikg2(cls, root: Path, out: Path, seed: int) -> None:
    #     """Build the OGB dataset into a simple .npz file, for faster loading."""
    #     data = ogb.linkproppred.LinkPropPredDataset("ogbl-wikikg2", root=root)
    #     split = data.get_edge_split()
    #     parts = {}
    #     random = np.random.default_rng(seed)
    #     for part in ["train", "valid"]:
    #         hrt = split[part]
    #         parts[part] = np.stack([hrt["head"], hrt["relation"], hrt["tail"]], axis=-1)
    #         random.shuffle(parts[part])
    #     np.savez(
    #         out,
    #         n_entity=int(data[0]["num_nodes"]),
    #         n_relation_type=int(1 + np.max(data.graph["edge_reltype"])),
    #         **{f"part_{k}": v for k, v in parts.items()},
    #     )

    # @classmethod
    # def load(cls, path: Path) -> "Dataset":
    #     """Load a dataset from an .npz file saved by `Dataset.build_wikikg2`."""
    #     data = np.load(path)
    #     return cls(
    #         n_entity=int(data["n_entity"]),
    #         n_relation_type=int(data["n_relation_type"]),
    #         triples={
    #             k.replace("part_", ""): data[k] for k in data if k.startswith("part_")
    #         },
    #     )

    # def save(self, path: Path) -> None:
    #     raise NotImplementedError

    # def shardpair_size(self, part: str, sharding: "Sharding") -> np.ndarray:
    #     """Measure shardpair sizes wrt sharding."""
    #     n_shard = sharding.n_shard
    #     shard_h, shard_t = sharding.entity_to_shard[self.triples[part][:, [0, 2]].T]
    #     shardpair_idx = shard_h * sharding.n_shard + shard_t
    #     shardpair_counts = np.bincount(
    #         shardpair_idx, minlength=n_shard * n_shard
    #     ).reshape(n_shard, n_shard)
    #     return shardpair_idx, shardpair_counts

    # def shard_triples(self, part: str, sharding: "Sharding") -> np.ndarray:
    #     """Divide triples in sharding.n_shard**2 buckets, based on head and tail shards."""
    #     sharpair_idx, shardpair_counts = self.shardpair_size(part, sharding)
    #     shardpair_offsets = np.concatenate(
    #         [[0], np.cumsum(shardpair_counts)[:-1]]
    #     ).reshape(sharding.n_shard, sharding.n_shard)
    #     sort_idx = np.argsort(sharpair_idx)
    #     # triple_sorted = self.triples[part][np.argsort(sharpair_idx)]
    #     # triple_sorted[:, [0,2]] = sharding.entity_to_idx[triple_sorted[:, [0,2]]]
    #     return shardpair_counts, shardpair_offsets, sort_idx

    # def shard_negatives(
    #     self, part: str, scheme: str, negative_size: int, sharding: "Sharding"
    # ) -> np.ndarray:
    #     if scheme == "h":
    #         negatives = self.neg_heads[part][:, :negative_size]
    #     elif scheme == "t":
    #         negatives = self.neg_tails[part][:, :negative_size]
    #     else:
    #         raise ValueError("scheme needs to be h / t")

    #     n_triple = negatives.shape[0]
    #     n_shard = sharding.n_shard
    #     shard_neg = sharding.entity_to_shard[negatives]
    #     shard_neg_counts = np.bincount(
    #         (shard_neg + n_shard * np.arange(n_triple)[:, None]).flatten(),
    #         minlength=n_shard * n_triple,
    #     ).reshape(n_triple, n_shard)
    #     shard_neg_offsets = np.c_[
    #         [0] * n_triple, np.cumsum(shard_neg_counts, axis=-1)[:, :-1]
    #     ]
    #     sort_neg_idx = np.argsort(shard_neg, axis=-1)
    #     return shard_neg_counts, shard_neg_offsets, sort_neg_idx


@dataclass
class Sharding:
    """A mapping of entities to shards (and back again).
    entity_to_shard -- int64[n_entity] -- maps entity ID to shard index
    entity_to_idx -- int64[n_entity] -- maps entity ID to index within its shard
    shard_and_idx_to_entity -- int64[n_shard, max_entity_per_shard]
                            -- maps [shard, idx] to entity ID
    """

    n_shard: int
    entity_to_shard: np.ndarray  # int64[n_entity]
    entity_to_idx: np.ndarray  # int64[n_entity]
    shard_and_idx_to_entity: np.ndarray  # int64[n_shard, max_entity_per_shard]
    shard_counts: int  # int64[n_shard] actual elements
    entity_type_counts: np.ndarray  # ints64[n_shard, n_types]
    entity_type_offsets: np.ndarray  # ints64[n_shard, n_types]

    @property
    def n_entity(self) -> int:
        return len(self.entity_to_shard)

    @property
    def max_entity_per_shard(self) -> int:
        return self.shard_and_idx_to_entity.shape[1]

    @classmethod
    def create(
        cls, n_entity: int, n_shard: int, seed: int, type_offsets: np.ndarray = None
    ) -> "Sharding":
        """Construct a random balanced assignment of entities to shards."""
        max_entity_per_shard = int(np.ceil(n_entity / n_shard))
        shard_and_idx_to_entity = np.sort(
            np.random.default_rng(seed)
            .permutation(n_shard * max_entity_per_shard)
            .reshape(n_shard, -1),
            axis=1,
        )
        entity_to_shard, entity_to_idx = np.divmod(
            np.argsort(shard_and_idx_to_entity.flatten())[:n_entity],
            max_entity_per_shard,
        )
        shard_deduction = shard_and_idx_to_entity[:, -1] >= n_entity
        shard_counts = max_entity_per_shard - shard_deduction
        if type_offsets is not None:
            type_id_per_shard = (
                np.digitize(shard_and_idx_to_entity, bins=type_offsets)
                + len(type_offsets) * np.arange(n_shard)[:, None]
                - 1
            )
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
