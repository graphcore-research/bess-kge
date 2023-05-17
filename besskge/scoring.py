# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional, Union, cast

import poptorch_experimental_addons as pea
import torch

from besskge.embedding import (
    EmbeddingInitializer,
    initialize_entity_embedding,
    initialize_relation_embedding,
    refactor_embedding_sharding,
)
from besskge.sharding import Sharding
from besskge.utils import complex_multiplication, complex_rotation


class BaseScoreFunction(torch.nn.Module, ABC):
    """
    Base class for scoring functions.
    """

    #: Share negative entities to construct negative samples
    negative_sample_sharing: bool

    #: Sharding of entities
    sharding: Sharding

    #: Relation embedding table
    entity_embedding: torch.nn.Parameter

    #: Relation embedding table
    relation_embedding: torch.nn.Parameter

    @abstractmethod
    def score_triple(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score (h,r,t) triples. No sharding is used.

        :param head_emb: shape: (batch_size, embedding_size)
            Embeddings of head entities in batch.
        :param relation_id: shape: (batch_size,)
            IDs of relation types in batch.
        :param tail_emb: shape: (batch_size, embedding_size)
            Embeddings of tail entities in batch.

        :return: shape: (batch_size,)
            Scores of batch triples.
        """
        raise NotImplementedError

    @abstractmethod
    def score_heads(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score sets of head entities against fixed (r,t) queries.

        :param head_emb: shape: (B, n_heads, embedding_size) with B = 1, batch_size
            Embeddings of head entities.
        :param relation_id: shape: (batch_size,)
            IDs of relation types in batch.
        :param tail_emb: shape: (batch_size, embedding_size)
            Embedding of tail entities in batch.

        :return: shape: (batch_size, B * n_heads)
            if :attr:`BaseScoreFunction.negative_sample_sharing`
            else (batch_size, n_heads).
            Scores of broadcasted triples.
        """
        raise NotImplementedError

    @abstractmethod
    def score_tails(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score sets of tail entities against fixed (h,r) queries.

        :param head_emb: shape: (batch_size, embedding_size)
            Embeddings of head entities in batch.
        :param relation_id: shape: (batch_size,)
            IDs of relation types in batch.
        :param tail_emb: shape: (B, n_tails, embedding_size) with B = 1, batch_size
            Embedding of tail entities.

        :return: shape: (batch_size, B * n_tails)
            if :attr:`BaseScoreFunction.negative_sample_sharing`
            else (batch_size, n_tails)
            Scores of broadcasted triples.
        """
        raise NotImplementedError

    def forward(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        see :meth:`BaseScoreFunction.score_triple`
        """
        return self.score_triple(head_emb, relation_id, tail_emb)

    def update_sharding(
        self,
        new_sharding: Sharding,
    ) -> None:
        """
        Change the sharding of the entity embedding table.

        :param new_sharding:
            The new entity sharding.
        """
        self.entity_embedding = refactor_embedding_sharding(
            entity_embedding=self.entity_embedding,
            old_sharding=self.sharding,
            new_sharding=new_sharding,
        )

        self.sharding = new_sharding


class DistanceBasedScoreFunction(BaseScoreFunction, ABC):
    """
    Base class for distance-based scoring functions.
    """

    # scoring_norm: int
    def __init__(self, negative_sample_sharing: bool, scoring_norm: int) -> None:
        """
        Initialize distance-based scoring function.

        :param negative_sample_sharing:
            see :class:`BaseScoreFunction`
        :param scoring_norm:
            p for p-norm to use in distance computation.
        """
        super().__init__()
        self.negative_sample_sharing = negative_sample_sharing
        self.scoring_norm = scoring_norm

    def reduce_embedding(self, v: torch.Tensor) -> torch.Tensor:
        """
        p-norm reduction along embedding dimension.

        :param v: shape: (*, embedding_size)
            The tensor to reduce.

        :return: shape: (*,)
            p-norm reduction.
        """
        return cast(torch.Tensor, torch.norm(v, p=self.scoring_norm, dim=-1))

    def broadcasted_score(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Broadcasted scores of queries against sets of entities.

        For each query and candidate, the score is given by the p-distance
        of the embeddings.

        :param v1: shape: (batch_size, embedding_size)
            Batch queries.
        :param v2: shape: (B, n_neg, embedding_size) with B = 1, batch_size
            Negative embeddings to score against queries.

        :return: shape: (batch_size, B * n_neg) if
                :attr:`BaseScoreFunction.negative_sample_sharing`
                else (batch_size, n_neg)
        """

        embedding_size = v1.shape[-1]
        if self.negative_sample_sharing:
            score = pea.distance_matrix(
                v1, v2.view(-1, embedding_size), p=self.scoring_norm
            )
        else:
            score = self.reduce_embedding(v1.unsqueeze(1) - v2)
        return cast(torch.Tensor, score)


class MatrixDecompositionScoreFunction(BaseScoreFunction, ABC):
    """
    Base class for matrix-decomposition scoring functions.
    """

    # scoring_norm: int
    def __init__(self, negative_sample_sharing: bool) -> None:
        """
        Initialize matrix-decomposition scoring function.

        :param negative_sample_sharing:
            see :class:`BaseScoreFunction`
        """
        super().__init__()
        self.negative_sample_sharing = negative_sample_sharing

    def reduce_embedding(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sum reduction along the embedding dimension.

        :param v: shape: (*, embedding_size)
            The tensor to reduce.

        :return: shape: (*,)
            Sum reduction.
        """
        return torch.sum(v, dim=-1)

    def broadcasted_score(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Broadcasted scores of queries against sets of entities.

        For each query and candidate, the score is given by the dot product of
        the embeddings.

        :param v1: shape: (batch_size, embedding_size)
            Batch queries.
        :param v2: shape: (B, n_neg, embedding_size) with B = 1, batch_size
            Negative embeddings to score against queries.

        :return: shape: (batch_size, B * n_neg) if
                :attr:`BaseScoreFunction.negative_sample_sharing`
                else (batch_size, n_neg)
        """

        embedding_size = v1.shape[-1]
        if self.negative_sample_sharing:
            score = torch.matmul(v1, v2.reshape(-1, embedding_size).T)
        else:
            score = self.reduce_embedding(v1.unsqueeze(1) * v2)
        return score


class TransE(DistanceBasedScoreFunction):
    """
    TransE scoring function :cite:p:`TransE`.
    """

    def __init__(
        self,
        negative_sample_sharing: bool,
        scoring_norm: int,
        sharding: Sharding,
        n_relation_type: int,
        embedding_size: Optional[int],
        entity_initializer: Union[torch.Tensor, EmbeddingInitializer],
        relation_initializer: Union[torch.Tensor, EmbeddingInitializer],
    ) -> None:
        """
        Initialize TransE model.

        :param negative_sample_sharing:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type scoring_norm:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type sharding:
            Entity sharding.
        :param n_relation_type:
            Number of relation types in the knowledge graph.
        :param embedding_size:
            Size of entities and relation embeddings. Can be omitted
            if passing tensors for initialization of entity and relation
            embeddings.
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(TransE, self).__init__(
            negative_sample_sharing=negative_sample_sharing, scoring_norm=scoring_norm
        )

        self.sharding = sharding

        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, embedding_size
        )
        assert (
            self.entity_embedding.shape[-1] == self.relation_embedding.shape[-1]
        ), "TransE requires the same embedding size for entities and relations"

    # docstr-coverage: inherited
    def score_triple(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return -self.reduce_embedding(head_emb + relation_emb - tail_emb)

    # docstr-coverage: inherited
    def score_heads(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return -self.broadcasted_score(tail_emb - relation_emb, head_emb)

    # docstr-coverage: inherited
    def score_tails(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return -self.broadcasted_score(head_emb + relation_emb, tail_emb)


class RotatE(DistanceBasedScoreFunction):
    """
    RotatE scoring function :cite:p:`RotatE`.
    """

    def __init__(
        self,
        negative_sample_sharing: bool,
        scoring_norm: int,
        sharding: Sharding,
        n_relation_type: int,
        embedding_size: int,
        entity_initializer: Union[torch.Tensor, EmbeddingInitializer],
        relation_initializer: Union[torch.Tensor, EmbeddingInitializer],
    ) -> None:
        """
        Initialize RotatE model.

        :param negative_sample_sharing:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type scoring_norm:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type sharding:
            Entity sharding.
        :param n_relation_type:
            Number of relation types in the knowledge graph.
        :param embedding_size:
            Size of entity embeddings (relation embedding size
            will be half of this). Can be omitted
            if passing tensors for initialization of entity and relation
            embeddings.
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(RotatE, self).__init__(
            negative_sample_sharing=negative_sample_sharing, scoring_norm=scoring_norm
        )

        self.sharding = sharding

        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, embedding_size // 2
        )
        assert (
            self.entity_embedding.shape[-1] % 2 == 0
        ), "RotatE requires an even real embedding size for entities"
        assert (
            self.entity_embedding.shape[-1] // 2 == self.relation_embedding.shape[-1]
        ), "RotatE requires relation embedding size to be half entity embedding size"

    # docstr-coverage: inherited
    def score_triple(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return -self.reduce_embedding(
            complex_rotation(head_emb, relation_emb) - tail_emb
        )

    # docstr-coverage: inherited
    def score_heads(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return -self.broadcasted_score(
            complex_rotation(tail_emb, -relation_emb), head_emb
        )

    # docstr-coverage: inherited
    def score_tails(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return -self.broadcasted_score(
            complex_rotation(head_emb, relation_emb), tail_emb
        )


class DistMult(MatrixDecompositionScoreFunction):
    """
    DistMult scoring function :cite:p:`DistMult`.
    """

    def __init__(
        self,
        negative_sample_sharing: bool,
        sharding: Sharding,
        n_relation_type: int,
        embedding_size: Optional[int],
        entity_initializer: Union[torch.Tensor, EmbeddingInitializer],
        relation_initializer: Union[torch.Tensor, EmbeddingInitializer],
    ) -> None:
        """
        Initialize DistMult model.

        :param negative_sample_sharing:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type sharding:
            Entity sharding.
        :param n_relation_type:
            Number of relation types in the knowledge graph.
        :param embedding_size:
            Size of entity and relation embeddings. Can be omitted
            if passing tensors for initialization of entity and relation
            embeddings.
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(DistMult, self).__init__(negative_sample_sharing=negative_sample_sharing)

        self.sharding = sharding

        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, embedding_size
        )
        assert (
            self.entity_embedding.shape[-1] == self.relation_embedding.shape[-1]
        ), "DistMult requires the same embedding size for entities and relations"

    # docstr-coverage: inherited
    def score_triple(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return self.reduce_embedding(head_emb * relation_emb * tail_emb)

    # docstr-coverage: inherited
    def score_heads(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return self.broadcasted_score(relation_emb * tail_emb, head_emb)

    # docstr-coverage: inherited
    def score_tails(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return self.broadcasted_score(head_emb * relation_emb, tail_emb)


class ComplEx(MatrixDecompositionScoreFunction):
    """
    ComplEx scoring function :cite:p:`ComplEx`.
    """

    def __init__(
        self,
        negative_sample_sharing: bool,
        sharding: Sharding,
        n_relation_type: int,
        embedding_size: Optional[int],
        entity_initializer: Union[torch.Tensor, EmbeddingInitializer],
        relation_initializer: Union[torch.Tensor, EmbeddingInitializer],
    ) -> None:
        """
        Initialize ComplEx model.

        :param negative_sample_sharing:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type sharding:
            Entity sharding.
        :param n_relation_type:
            Number of relation types in the knowledge graph.
        :param embedding_size:
            Size of entity and relation embeddings. Can be omitted
            if passing tensors for initialization of entity and relation
            embeddings.
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(ComplEx, self).__init__(negative_sample_sharing=negative_sample_sharing)

        self.sharding = sharding

        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, embedding_size
        )
        assert (
            self.entity_embedding.shape[-1] == self.relation_embedding.shape[-1]
        ), "ComplEx requires the same embedding size for entities and relations"
        assert (
            self.entity_embedding.shape[-1] % 2 == 0
        ), "ComplEx requires an even real embedding size for entities and relations"

    # docstr-coverage: inherited
    def score_triple(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return self.reduce_embedding(
            complex_multiplication(head_emb, relation_emb) * tail_emb
        )

    # docstr-coverage: inherited
    def score_heads(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        cutpoint = relation_emb.shape[-1] // 2
        relation_emb[:, cutpoint:] = -relation_emb[:, cutpoint:]  # conjugate relations
        return self.broadcasted_score(
            complex_multiplication(relation_emb, tail_emb), head_emb
        )

    # docstr-coverage: inherited
    def score_tails(
        self,
        head_emb: torch.Tensor,
        relation_id: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        relation_emb = torch.index_select(
            self.relation_embedding, index=relation_id, dim=0
        )
        return self.broadcasted_score(
            complex_multiplication(head_emb, relation_emb), tail_emb
        )
