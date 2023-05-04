# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch

from besskge.sharding import Sharding

"""
Utilities for entity/relation embeddings.
"""


class EmbeddingInitializer(ABC):
    """
    Base class for custom embedding initialization scheme.
    """

    @abstractmethod
    def initialize(self, embedding_table: torch.nn.Parameter) -> None:
        """
        Initialize embedding table.

        :param embedding_table:
            The embedding table to initialize (in-place)
        """
        raise NotImplementedError


class UniformInitializer(EmbeddingInitializer):
    def __init__(self, range_scale: float = 1.0):
        """
        Initialize embeddings according to uniform distribution
        in the range `[-range_scale / embedding_size, + range_scale / embedding_size]`.

        :param range_scale:
            Scaling factor for the range.
        """
        self.range_scale = range_scale

    # docstr-coverage: inherited
    def initialize(self, embedding_table: torch.nn.Parameter) -> None:
        embedding_range = self.range_scale / embedding_table.shape[-1]
        torch.nn.init.uniform_(embedding_table, -embedding_range, embedding_range)


class NormalInitializer(EmbeddingInitializer):
    def __init__(self, std_scale: float = 1.0):
        """
        Initialize embeddings according to normal distribution with
        mean 0 and standard deviation `std_scale / embedding_size`.

        :param std_scale:
            Scaling factor for standard deviation.
        """
        self.std_scale = std_scale

    # docstr-coverage: inherited
    def initialize(self, embedding_table: torch.nn.Parameter) -> None:
        std = self.std_scale / embedding_table.shape[-1]
        torch.nn.init.normal_(embedding_table, std=std)


def initialize_entity_embedding(
    initializer: Union[torch.Tensor, EmbeddingInitializer],
    sharding: Sharding,
    embedding_size: Optional[int] = None,
) -> torch.nn.Parameter:
    """
    Initialize entity embedding table.

    :param initializer:
        Embedding table or embedding initializer. If providing
        embedding table, this can either be sharded
        (shape: (n_shard, max_entity_per_shard, embedding_size))
        or unsharded (shape: (n_entity, embedding_size)).
    :param sharding:
        Entity sharding.
    :param embedding_size:
        Entity embedding size. Can be omitted if passing
        embedding table.

    :return: shape: (n_shard, max_ent_per_shard, embedding_size)
        Entity embedding table.
    """

    if isinstance(initializer, torch.Tensor):
        if initializer.dim() == 3:
            if initializer.size()[:2] != torch.Size(
                [sharding.n_shard, sharding.max_entity_per_shard]
            ):
                raise ValueError(
                    "Shape of sharded table provided for initialization"
                    " is not compatible with sharding"
                )
            entity_embedding = torch.nn.Parameter(initializer.to(torch.float32))
        elif initializer.dim() == 2:
            if initializer.shape[0] != sharding.n_entity:
                raise ValueError(
                    "Number of rows of table provided for initialization"
                    " different from number of entities."
                )
            initializer_sharded = initializer[
                torch.from_numpy(
                    np.minimum(sharding.shard_and_idx_to_entity, sharding.n_entity - 1)
                )
            ]
            entity_embedding = torch.nn.Parameter(initializer_sharded.to(torch.float32))
        else:
            raise ValueError("Table for initialization needs to be 2- or 3-dimensional")

        if embedding_size:
            assert (
                embedding_size == entity_embedding.shape[-1]
            ), "Initialization tensor and embedding_size provided are incompatible"
    else:
        if not embedding_size:
            raise ValueError(
                "If not providing embedding table, embedding_size needs to be specified"
            )
        entity_embedding = torch.nn.Parameter(
            torch.empty(
                size=(
                    sharding.n_shard,
                    sharding.max_entity_per_shard,
                    embedding_size,
                ),
                dtype=torch.float32,
            )
        )
        initializer.initialize(entity_embedding)

    return entity_embedding


def initialize_relation_embedding(
    initializer: Union[torch.Tensor, EmbeddingInitializer],
    n_relation_type: int,
    embedding_size: Optional[int] = None,
) -> torch.nn.Parameter:
    """
    Initialize relation embedding table.

    :param initializer:
        Embedding table or embedding initializer.
    :param n_relation_type:
        Number of relation types.
    :param embedding_size:
        Relation embedding size. Can be omitted if passing
        embedding table.

    :return:
        Relation embedding table.
    """

    if isinstance(initializer, torch.Tensor):
        if initializer.dim() != 2:
            raise ValueError("Table for initialization needs to be 2-dimensional")
        relation_embedding = torch.nn.Parameter(initializer.to(torch.float32))

        if embedding_size:
            assert (
                embedding_size == relation_embedding.shape[-1]
            ), "Initialization tensor and embedding_size provided are incompatible"
    else:
        if not embedding_size:
            raise ValueError(
                "If not providing embedding table, embedding_size needs to be specified"
            )
        relation_embedding = torch.nn.Parameter(
            torch.empty(size=(n_relation_type, embedding_size), dtype=torch.float32)
        )
        initializer.initialize(relation_embedding)

    return relation_embedding


def refactor_embedding_sharding(
    entity_embedding: torch.nn.Parameter,
    old_sharding: Sharding,
    new_sharding: Sharding,
) -> torch.nn.Parameter:
    """
    Refactor sharded entity embedding table to pass from
    one entity sharding to a different one.

    :param entity_embedding: shape: (n_shard_old, max_ent_per_shard_old, embedding_size)
        Entity embedding table sharded according to old_sharding.
    :param old_sharding:
        The current entity sharding.
    :param new_sharding:
        The new entity sharding.

    :return: shape: (n_shard_new, max_ent_per_shard_new, embedding_size)
        The refactored entity embedding table, sharded according
        to new_sharding.
    """

    embedding_table = entity_embedding.detach()
    unsharded_table = embedding_table[
        old_sharding.entity_to_shard, old_sharding.entity_to_idx
    ]

    return initialize_entity_embedding(
        initializer=unsharded_table, sharding=new_sharding
    )
