# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable, List, Optional, Union

import numpy as np
import torch

from besskge.sharding import Sharding

"""
Utilities for entity/relation embeddings.
"""


def init_uniform_norm(embedding_table: torch.Tensor) -> torch.Tensor:
    """
    Initialize embeddings according to uniform distribution
    and normalize so that each row has norm 1.

    :param embedding_table:
        Tensor of embedding parameters to initialize.

    :return:
        Initialized tensor.
    """
    return torch.nn.functional.normalize(torch.nn.init.uniform(embedding_table), dim=-1)


def init_KGE_uniform(
    embedding_table: torch.Tensor, b: float = 1.0, divide_by_embedding_size: bool = True
) -> torch.Tensor:
    """
    Initialize embeddings according to symmetric uniform distribution.

    :param embedding_table:
        Tensor of embedding parameters to initialize.
    :param b:
        Positive boundary of distribution support. Default: 1.0.
    :param divide_by_embedding_size:
        Rescale distribution support by `1/row_size`. Default: True.

    :return:
        Initialized tensor.
    """
    if divide_by_embedding_size:
        b /= embedding_table.shape[-1]
    return torch.nn.init.uniform_(embedding_table, -b, b)


def init_KGE_normal(
    embedding_table: torch.Tensor,
    std: float = 1.0,
    divide_by_embedding_size: bool = True,
) -> torch.Tensor:
    """
    Initialize embeddings according to normal distribution with mean 0.

    :param embedding_table:
        Tensor of embedding parameters to initialize.
    :param std:
        Standard deviation. Default: 1.0.
    :param divide_by_embedding_size:
        Rescale standard deviation by `1/row_size`. Default: True.

    :return:
        Initialized tensor.
    """
    if divide_by_embedding_size:
        std /= embedding_table.shape[-1]
    return torch.nn.init.normal_(embedding_table, std=std)


def initialize_entity_embedding(
    sharding: Sharding,
    initializer: Union[torch.Tensor, List[Callable[..., torch.Tensor]]],
    row_size: Optional[List[int]] = None,
) -> torch.nn.Parameter:
    """
    Initialize entity embedding table.

    :param sharding:
        Entity sharding.
    :param initializer:
        Embedding table or list of initializing functions. If providing
        an embedding table, this can either be sharded
        (shape: [n_shard, max_entity_per_shard, row_size])
        or unsharded [shape: (n_entity, row_size]).
        If providing list of initializers, this needs to be of same length
        as :attr:`row_size`.
    :param row_size:
        Number of parameters for each entity.
        This needs to be a list, with the lengths of the different embedding tensors
        to allocate for each entity. Each embedding tensor, once allocated, is
        initialized with the corresponding entry of :attr:`initializer`.
        Can be omitted if passing an embedding table as :attr:`initializer`.

    :return: shape: (n_shard, max_ent_per_shard, row_size)
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
            entity_embedding = initializer.to(torch.float32)
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
            entity_embedding = initializer_sharded.to(torch.float32)
        else:
            raise ValueError("Table for initialization needs to be 2- or 3-dimensional")

        if row_size:
            assert (
                sum(row_size) == entity_embedding.shape[-1]
            ), "Initialization tensor and row_size provided are incompatible"
    else:
        if not row_size:
            raise ValueError(
                "If not providing an embedding table, row_size needs to be specified"
            )
        if len(initializer) != len(row_size):
            raise ValueError(
                "Different number of embedding splits and initializers provided"
            )
        entity_embedding = torch.empty(
            (sharding.n_shard, sharding.max_entity_per_shard, 0),
            dtype=torch.float32,
        )
        for slice_size, init in zip(row_size, initializer):
            table_slice = init(
                torch.empty(
                    size=(
                        sharding.n_shard,
                        sharding.max_entity_per_shard,
                        slice_size,
                    ),
                    dtype=torch.float32,
                )
            )
            entity_embedding = torch.concat([entity_embedding, table_slice], dim=-1)

    return torch.nn.Parameter(entity_embedding)


def initialize_relation_embedding(
    n_relation_type: int,
    inverse_relations: bool,
    initializer: Union[torch.Tensor, List[Callable[..., torch.Tensor]]],
    row_size: Optional[List[int]] = None,
) -> torch.nn.Parameter:
    """
    Initialize relation embedding table.

    :param n_relation_type:
        Number of relation types.
    :param inverse_relations:
        If True, learn embeddings for inverse relations, in addition to direct ones.
        Needs to be set to `True` when inverse triples are added to the dataset.
        Given a relation with ID `i`, its inverse is the one with
        ID `i+n_relation_type`.
    :param initializer:
         Embedding table or list of initializing functions.
         If providing list of initializers, this needs to be of same length
         as :attr:`row_size`.
    :param row_size:
        Number of parameters for each relation type.
        This needs to be a list, with the lengths of the different embedding tensors
        to allocate for each relation. Each embedding tensor, once allocated, is
        initialized with the corresponding entry of :attr:`initializer`.
        Can be omitted if passing an embedding table as :attr:`initializer`.

    :return:
        Relation embedding table.
    """

    if isinstance(initializer, torch.Tensor):
        if initializer.dim() != 2:
            raise ValueError("Table for initialization needs to be 2-dimensional")
        relation_embedding = initializer.to(torch.float32)

        if row_size:
            assert (
                sum(row_size) == relation_embedding.shape[-1]
            ), "Initialization tensor and row_size provided are incompatible"
    else:
        if not row_size:
            raise ValueError(
                "If not providing an embedding table, row_size needs to be specified"
            )
        if len(initializer) != len(row_size):
            raise ValueError(
                "Different number of embedding splits and initializers provided"
            )
        n_rows = 2 * n_relation_type if inverse_relations else n_relation_type
        relation_embedding = torch.empty(
            (n_rows, 0),
            dtype=torch.float32,
        )
        for slice_size, init in zip(row_size, initializer):
            table_slice = init(
                torch.empty(
                    size=(
                        n_rows,
                        slice_size,
                    ),
                    dtype=torch.float32,
                )
            )
            relation_embedding = torch.concat([relation_embedding, table_slice], dim=-1)

    return torch.nn.Parameter(relation_embedding)


def refactor_embedding_sharding(
    entity_embedding: torch.nn.Parameter,
    old_sharding: Sharding,
    new_sharding: Sharding,
) -> torch.nn.Parameter:
    """
    Refactor sharded entity embedding table to pass from
    one entity sharding to a different one.

    :param entity_embedding: shape: (n_shard_old, max_ent_per_shard_old, row_size)
        Entity embedding table sharded according to `old_sharding`.
    :param old_sharding:
        The current entity sharding.
    :param new_sharding:
        The new entity sharding.

    :return: shape: (n_shard_new, max_ent_per_shard_new, row_size)
        The refactored entity embedding table, sharded according
        to `new_sharding`.
    """

    embedding_table = entity_embedding.detach()
    unsharded_table = embedding_table[
        old_sharding.entity_to_shard, old_sharding.entity_to_idx
    ]

    return initialize_entity_embedding(
        initializer=unsharded_table, sharding=new_sharding
    )
