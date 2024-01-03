# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Utilities for building and storing knowledge graph datasets
as collections of (h,r,t) triples.
"""

import dataclasses
import pickle
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    #: Number of entities (nodes) in the knowledge graph
    n_entity: int

    #: Number of relation types (edge labels) in the knowledge graph
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
        Build a dataset from an array of triples, where IDs for entities
        and relations have already been assigned. Note that, if entities have
        types, entities of the same type need to have contiguous IDs.
        Triples are randomly split in train/validation/test sets.
        The attribute `KGDataset.original_triple_ids` stores the IDs
        of the triples in each split wrt the original ordering in `data`.

        If a pre-defined train/validation/test split is wanted, the KGDataset
        class should be instantiated manually.

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
        id_shuffle = rng.permutation(np.arange(num_triples))
        triple_ids = dict()
        triple_ids["train"], triple_ids["valid"], triple_ids["test"] = np.split(
            id_shuffle, (num_train, num_train + num_valid), axis=0
        )
        triples = dict()
        for split in ["train", "valid", "test"]:
            triples[split] = data[triple_ids[split]]

        ds = cls(
            n_entity=data[:, [0, 2]].max() + 1,
            n_relation_type=data[:, 1].max() + 1,
            entity_dict=entity_dict,
            relation_dict=relation_dict,
            type_offsets=type_offsets,
            triples=triples,
        )
        ds.original_triple_ids = triple_ids

        return ds

    @classmethod
    def from_dataframe(
        cls,
        df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        head_column: Union[int, str],
        relation_column: Union[int, str],
        tail_column: Union[int, str],
        entity_types: Optional[Union[pd.Series, Dict[str, str]]] = None,  # type: ignore
        split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 1234,
    ) -> "KGDataset":
        """
        Build a KGDataset from a pandas DataFrame of labeled (h,r,t) triples.
        IDs for entities and relations are automatically assigned based on labels
        in such a way that entities of the same type have contiguous IDs.

        :param df:
            Pandas DataFrame of all triples in the knowledge graph dataset,
            or dictionary of DataFrames of triples for each part of the dataset split
        :param head_column:
            Name of the DataFrame column storing head entities
        :param relation_column:
            Name of the DataFrame column storing relations
        :param tail_column:
            Name of the DataFrame column storing tail entities
        :param entity_types:
            If entities have types, dictionary or pandas Series of mappings
            entity label -> entity type (as strings).
        :param split:
            Tuple to set the train/validation/test split.
            Only used if no pre-defined dataset split is specified,
            i.e. if `df` is not a dictionary.
        :param seed:
            Random seed for the train/validation/test split.
            Only used if no pre-defined dataset split is specified,
            i.e. if `df` is not a dictionary.

        :return: Instance of the KGDataset class.
        """

        df_dict = {"all": df} if isinstance(df, pd.DataFrame) else df
        unique_ent = pd.concat(
            [
                pd.concat([dfp[head_column], dfp[tail_column]])
                for dfp in df_dict.values()
            ]
        ).unique()
        ent2id = pd.Series(np.arange(len(unique_ent)), index=unique_ent, name="ent_id")
        unique_rel = pd.concat(
            [dfp[relation_column] for dfp in df_dict.values()]
        ).unique()
        rel2id = pd.Series(np.arange(len(unique_rel)), index=unique_rel, name="rel_id")

        if entity_types is not None:
            ent2type = pd.Series(entity_types, name="ent_type")
            ent2id_type = pd.merge(
                ent2id, ent2type, how="left", left_index=True, right_index=True
            ).sort_values("ent_type")
            ent2id.index = ent2id_type.index
            type_off = (
                ent2id_type.groupby("ent_type")["ent_type"].count().cumsum().shift(1)
            )
            type_off.iloc[0] = 0
            type_offsets = type_off.astype("int64").to_dict()
        else:
            type_offsets = None

        entity_dict = ent2id.index.tolist()
        relation_dict = rel2id.index.tolist()

        triples = {}
        for part, dfp in df_dict.items():
            heads = dfp[head_column].map(ent2id).values.astype(np.int32)
            tails = dfp[tail_column].map(ent2id).values.astype(np.int32)
            rels = dfp[relation_column].map(rel2id).values.astype(np.int32)
            triples[part] = np.stack([heads, rels, tails], axis=1)

        if isinstance(df, pd.DataFrame):
            return KGDataset.from_triples(
                triples["all"], split, seed, entity_dict, relation_dict, type_offsets
            )
        else:
            return cls(
                n_entity=len(entity_dict),
                n_relation_type=len(relation_dict),
                entity_dict=entity_dict,
                relation_dict=relation_dict,
                type_offsets=type_offsets,
                triples=triples,
            )

    @classmethod
    def build_ogbl_biokg(cls, root: Path) -> "KGDataset":
        """
        Build the ogbl-biokg dataset :cite:p:`OGB`

        .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg

        :param root:
            Local path to the dataset. If the dataset is not present in this
            location, then it is downloaded and stored here.

        :return: The ogbl-biokg KGDataset.
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
    def build_ogbl_wikikg2(cls, root: Path) -> "KGDataset":
        """
        Build the ogbl-wikikg2 dataset :cite:p:`OGB`

        .. seealso:: https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2

        :param root:
            Local path to the dataset. If the dataset is not present in this
            location, then it is downloaded and stored here.

        :return: The ogbl-wikikg2 KGDataset.
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
        This is the subgraph of the YAGO3 knowledge
        graph :cite:p:`YAGO3` containing only entities which have at least 10
        relations associated to them. First used in :cite:p:`ConvE`.

        .. seealso:: https://yago-knowledge.org/downloads/yago-3

        :param root:
            Local path to the dataset. If the dataset is not present in this
            location, then it is downloaded and stored here.

        :return: The YAGO3-10 KGDataset.
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

        train_triples = pd.read_csv(
            root.joinpath("train.txt"), delimiter="\t", dtype=str, header=None
        )
        valid_triples = pd.read_csv(
            root.joinpath("valid.txt"), delimiter="\t", dtype=str, header=None
        )
        test_triples = pd.read_csv(
            root.joinpath("test.txt"), delimiter="\t", dtype=str, header=None
        )

        return cls.from_dataframe(
            {"train": train_triples, "valid": valid_triples, "test": test_triples},
            head_column=0,
            relation_column=1,
            tail_column=2,
        )

    @classmethod
    def build_openbiolink(cls, root: Path) -> "KGDataset":
        """
        Build the high-quality version of the OpenBioLink2020
        dataset :cite:p:`openbiolink`

        .. seealso:: https://github.com/openbiolink/openbiolink#benchmark-dataset

        :param root:
            Local path to the dataset. If the dataset is not present in this
            location, then it is downloaded and stored here.

        :return: The HQ OpenBioLink2020 KGDataset.
        """

        if not (
            root.joinpath("HQ_DIR/train_test_data/train_sample.csv").is_file()
            and root.joinpath("HQ_DIR/train_test_data/val_sample.csv").is_file()
            and root.joinpath("HQ_DIR/train_test_data/test_sample.csv").is_file()
            and root.joinpath("HQ_DIR/train_test_data/train_val_nodes.csv").is_file()
        ):
            print("Downloading dataset...")
            res = requests.get(url="https://zenodo.org/record/3834052/files/HQ_DIR.zip")
            with zipfile.ZipFile(BytesIO(res.content)) as zip_f:
                zip_f.extractall(path=root)

        column_names = ["h_label", "r_label", "t_label", "quality", "TP/TN", "source"]
        train_triples = pd.read_csv(
            root.joinpath("HQ_DIR/train_test_data/train_sample.csv"),
            header=None,
            names=column_names,
            sep="\t",
        )
        valid_triples = pd.read_csv(
            root.joinpath("HQ_DIR/train_test_data/val_sample.csv"),
            header=None,
            names=column_names,
            sep="\t",
        )
        test_triples = pd.read_csv(
            root.joinpath("HQ_DIR/train_test_data/test_sample.csv"),
            header=None,
            names=column_names,
            sep="\t",
        )

        entity_types = pd.read_csv(
            root.joinpath("HQ_DIR/train_test_data/train_val_nodes.csv"),
            header=None,
            names=["ent_label", "ent_type"],
            sep="\t",
        ).set_index("ent_label")["ent_type"]

        return cls.from_dataframe(
            {"train": train_triples, "valid": valid_triples, "test": test_triples},
            head_column="h_label",
            relation_column="r_label",
            tail_column="t_label",
            entity_types=entity_types,
        )

    def save(self, out_file: Path) -> None:
        """
        Save dataset to .pkl.

        :param out_file:
            Path to output file.
        """
        with open(out_file, "wb") as f:
            pickle.dump(self, f)
        print(f"KGDataset saved to {out_file}")

    @classmethod
    def load(cls, path: Path) -> "KGDataset":
        """
        Load a :class:`KGDataset` object saved with :func:`KGDataset.save`.

        :param path:
            Path to saved :class:`KGDataset` object.

        :return:
            The saved :class:`KGDataset` object.
        """
        kg_dataset: KGDataset
        with open(path, "rb") as f:
            kg_dataset = pickle.load(f)
            if not isinstance(kg_dataset, KGDataset):
                raise ValueError(f"File at path {path} is not a KGDataset")
        print(f"Loaded KGDataset at {path}")

        return kg_dataset
