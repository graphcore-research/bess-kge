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
