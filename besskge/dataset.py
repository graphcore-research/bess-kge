# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import dataclasses
import pickle
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ogb.linkproppred
import pandas as pd
import requests
from numpy.typing import NDArray


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
    #: {part: int32[n_triple, {h,r,t}]}
    triples: Dict[str, NDArray[np.int32]]

    #: Entity labels by ID; str[n_entity]
    entity_dict: Optional[List[str]] = None

    #: Relation type labels by ID; str[n_relation_type]
    relation_dict: Optional[List[str]] = None

    #: If entities have types, IDs are assumed to be clustered by type;
    #: {entity_type: int}
    type_offsets: Optional[Dict[str, int]] = None

    #: IDs of (possibly triple-specific) negative heads;
    #: {part: int32[n_triple or 1, n_neg_heads]}
    neg_heads: Optional[Dict[str, NDArray[np.int32]]] = None

    #: IDs of (possibly triple-specific) negative tails;
    #: {part: int32[n_triple or 1, n_neg_tails]}
    neg_tails: Optional[Dict[str, NDArray[np.int32]]] = None

    @property
    def ht_types(self) -> Optional[Dict[str, NDArray[np.int32]]]:
        """
        If entities have types, type IDs of triples' heads/tails;
        {part: int32[n_triple, {h_type, t_type}]}
        """
        if self.type_offsets:
            type_offsets = np.fromiter(self.type_offsets.values(), dtype=np.int32)
            types = {}
            for part, triple in self.triples.items():
                types[part] = (
                    np.digitize(
                        triple[:, [0, 2]],
                        type_offsets,
                    )
                    - 1
                )
            return types
        else:
            return None

    @classmethod
    def build_biokg(cls, root: Path) -> "KGDataset":
        """
        Build the ogbl-biokg dataset :cite:p:`OGB`

        .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg

        :param root:
            Path to dataset. If dataset is not present, download it
            at this path.

        :return: ogbl-biokg KGDataset.
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
        neg_heads = {}
        neg_tails = {}
        for part, hrt in split_edge.items():
            h_label, h_type_idx = np.unique(hrt["head_type"], return_inverse=True)
            t_label, t_type_idx = np.unique(hrt["tail_type"], return_inverse=True)
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

        ent_dict: List[str] = []
        for k in type_offsets.keys():
            ent_dict.extend(
                pd.read_csv(root.joinpath(f"ogbl_biokg/mapping/{k}_entidx2name.csv.gz"))
                .sort_values("ent idx")["ent name"]
                .values.tolist()
            )
        rel_dict = (
            pd.read_csv(root.joinpath("ogbl_biokg/mapping/relidx2relname.csv.gz"))
            .sort_values("rel idx")["rel name"]
            .values.tolist()
        )

        return cls(
            n_entity=n_entity,
            n_relation_type=n_relation_type,
            entity_dict=ent_dict,
            relation_dict=rel_dict,
            type_offsets=type_offsets,
            triples=triples,
            neg_heads=neg_heads,
            neg_tails=neg_tails,
        )

    @classmethod
    def build_wikikg2(cls, root: Path) -> "KGDataset":
        """
        Build the ogbl-wikikg2 dataset :cite:p:`OGB`

        .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2

        :param root:
            Path to dataset. If dataset is not present, download it
            at this path.

        :return: ogbl-wikikg2 KGDataset.
        """
        dataset = ogb.linkproppred.LinkPropPredDataset(name="ogbl-wikikg2", root=root)
        split_data = dataset.get_edge_split()

        triples = {}
        neg_heads = {}
        neg_tails = {}
        for part, hrt in split_data.items():
            triples[part] = np.stack(
                [hrt["head"], hrt["relation"], hrt["tail"]], axis=-1
            )
            if part != "train":
                neg_heads[part] = hrt["head_neg"]
                neg_tails[part] = hrt["tail_neg"]

        ent_dict = (
            pd.read_csv(root.joinpath("ogbl_wikikg2/mapping/nodeidx2entityid.csv.gz"))
            .sort_values("node idx")["entity id"]
            .values.tolist()
        )
        rel_dict = (
            pd.read_csv(root.joinpath("ogbl_wikikg2/mapping/reltype2relid.csv.gz"))
            .sort_values("reltype")["rel id"]
            .values.tolist()
        )

        return cls(
            n_entity=dataset.graph["num_nodes"],
            n_relation_type=split_data["train"]["relation"].max() + 1,
            entity_dict=ent_dict,
            relation_dict=rel_dict,
            type_offsets=None,
            triples=triples,
            neg_heads=neg_heads,
            neg_tails=neg_tails,
        )

    @classmethod
    def build_yago310(cls, root: Path) -> "KGDataset":
        """
        Build the YAGO3-10 dataset.
        This is the subgraph of the YAGO3 KG :cite:p:`YAGO3` containing only
        entities which have at least 10 relations associated to them.
        First used in :cite:p:`ConvE`.

        .. seealso:: https://yago-knowledge.org/downloads/yago-3

        :param root:
            Path to dataset. If dataset is not present, download it
            at this path.

        :return: YAGO3-10 KGDataset.
        """

        if not (
            root.joinpath("train.txt").is_file()
            and root.joinpath("valid.txt").is_file()
            and root.joinpath("test.txt").is_file()
        ):
            print("Downloading dataset...")
            res = requests.get(
                url="https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz"
            )
            with tarfile.open(fileobj=BytesIO(res.content)) as tarf:
                tarf.extractall(path=root)

        train = np.loadtxt(root.joinpath("train.txt"), delimiter="\t", dtype=str)
        valid = np.loadtxt(root.joinpath("valid.txt"), delimiter="\t", dtype=str)
        test = np.loadtxt(root.joinpath("test.txt"), delimiter="\t", dtype=str)

        entity_dict, entity_id = np.unique(
            np.concatenate(
                [
                    train[:, 0],
                    train[:, 2],
                    valid[:, 0],
                    valid[:, 2],
                    test[:, 0],
                    test[:, 2],
                ]
            ),
            return_inverse=True,
        )
        entity_split_limits = np.cumsum(
            [
                train.shape[0],
                train.shape[0],
                valid.shape[0],
                valid.shape[0],
                test.shape[0],
            ]
        )
        (
            train_head_id,
            train_tail_id,
            validation_head_id,
            validation_tail_id,
            test_head_id,
            test_tail_id,
        ) = np.split(entity_id, entity_split_limits)

        rel_dict, rel_id = np.unique(
            np.concatenate([train[:, 1], valid[:, 1], test[:, 1]]),
            return_inverse=True,
        )
        relation_split_limits = np.cumsum([train.shape[0], valid.shape[0]])
        train_rel_id, validation_rel_id, test_rel_id = np.split(
            rel_id, relation_split_limits
        )

        triples = {
            "train": np.concatenate(
                [train_head_id[:, None], train_rel_id[:, None], train_tail_id[:, None]],
                axis=1,
            ),
            "validation": np.concatenate(
                [
                    validation_head_id[:, None],
                    validation_rel_id[:, None],
                    validation_tail_id[:, None],
                ],
                axis=1,
            ),
            "test": np.concatenate(
                [test_head_id[:, None], test_rel_id[:, None], test_tail_id[:, None]],
                axis=1,
            ),
        }

        return cls(
            n_entity=len(entity_dict),
            n_relation_type=len(rel_dict),
            entity_dict=entity_dict.tolist(),
            relation_dict=rel_dict.tolist(),
            type_offsets=None,
            triples=triples,
            neg_heads=None,
            neg_tails=None,
        )

    @classmethod
    def from_triples(
        cls,
        data: NDArray[np.int32],
        split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 1234,
        entity_dict: Optional[List[str]] = None,
        relation_dict: Optional[List[str]] = None,
        type_offsets: Optional[Dict[str, int]] = None,
    ) -> "KGDataset":
        """
        Build a dataset from an array of triples. Note that if a pre-defined
        train/validation/test split is wanted the KGDataset class should be instantiated
        manually.

        :param data:
            Numpy array of triples [head_id, relation_id, tail_id]. Shape
            (num_triples, 3).
        :param split:
            Tuple to set the train/validation/test split.
        :param seed:
            Random seed for the train/validation/test split.
        :param entity_dict:
            Optional entity labels by ID.
        :param relation_dict:
            Optional relation labels by ID.
        :param type_offsets:
            Offset of entity types

        :return: Instance of the KGDataset class.
        """
        num_triples = data.shape[0]
        num_train = int(num_triples * split[0])
        num_valid = int(num_triples * split[1])

        rng = np.random.default_rng(seed=seed)
        rng.shuffle(data, axis=0)

        triples = dict()
        triples["train"], triples["valid"], triples["test"] = np.split(
            data, (num_train, num_train + num_valid), axis=0
        )

        return cls(
            n_entity=data[:, [0, 2]].max() + 1,
            n_relation_type=data[:, 1].max() + 1,
            entity_dict=entity_dict,
            relation_dict=relation_dict,
            type_offsets=type_offsets,
            triples=triples,
        )

    def save(self, out_file: Path) -> None:
        """
        Save dataset to .pkl.

        :param out_file:
            Path to output file.
        """
        with open(out_file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "KGDataset":
        """
        Load a :class:`KGDataset` saved with :func:`KGDataset.save`.

        :param path:
            Path to saved :class:`KGDataset`.

        :return:
            The saved :class:`KGDataset`.
        """
        kg_dataset: KGDataset
        with open(path, "rb") as f:
            kg_dataset = pickle.load(f)
            if not isinstance(kg_dataset, KGDataset):
                raise ValueError(f"File at path {path} is not a KGDataset")

        return kg_dataset
