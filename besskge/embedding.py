from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import torch

from besskge.sharding import Sharding

"""
Utilities for entity/relation embeddings.
"""


class EmbeddingInitializer:
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


class MarginBasedInitializer(EmbeddingInitializer):
    def __init__(self, margin: float):
        """
        Margin-based initialization scheme.

        :param margin:
            The margin used in the loss function.
        """
        self.margin = margin

    # docstr-coverage: inherited
    def initialize(self, embedding_table: torch.nn.Parameter) -> None:
        embedding_range = self.margin / embedding_table.shape[-1]
        torch.nn.init.uniform_(embedding_table, -embedding_range, embedding_range)


def initialize_entity_embedding(
    initializer: Union[torch.Tensor, EmbeddingInitializer],
    sharding: Sharding,
    embedding_size: Optional[int] = None,
) -> torch.nn.Parameter:
    """
    Initialize entity embedding table.

    :param initializer:
        Embedding table or embedding intializer. If providing
        embedding table, this can either be sharded
        (shape: (n_shard, max_entity_per_shard, embedding_size))
        or unsharded (shape: (n_entity, embedding_size)).
    :param sharding:
        Entity sharding.
    :param embedding_size:
        Entity embedding size. Can be omitted if passing
        embedding table.

    :return:
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
            intializer_sharded = initializer[
                torch.from_numpy(
                    np.minimum(sharding.shard_and_idx_to_entity, sharding.n_entity - 1)
                )
            ]
            entity_embedding = torch.nn.Parameter(intializer_sharded.to(torch.float32))
        else:
            raise ValueError("Table for initialization needs to be 2- or 3-dimensional")
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
    embedding_size: int,
) -> torch.nn.Parameter:
    """
    Initialize relation embedding table.

    :param initializer:
        Embedding table or embedding intializer.
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
