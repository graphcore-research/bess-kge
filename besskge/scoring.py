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
        Score a batch of (h,r,t) triples.

        :param head_emb: shape: (batch_size, embedding_size)
            Embeddings of head entities in batch.
        :param relation_id: shape: (batch_size,)
            IDs of relation types in batch.
        :param tail_emb: shape: (batch_size, embedding_size)
            Embeddings of tail entities in batch.

        :return: shape: (batch_size,)
            Scores of a batch of triples.
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
        embedding_size: int,
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
            Size of entities and relation embeddings.
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
            self.entity_embedding.shape[-1]
            == self.relation_embedding.shape[-1]
            == embedding_size
        ), (
            "TransE requires `embedding_size` embedding parameters"
            " for each entity and relation"
        )
        self.embedding_size = embedding_size

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
            Complex size of entity embeddings (and real size of
            relation embeddings).
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(RotatE, self).__init__(
            negative_sample_sharing=negative_sample_sharing, scoring_norm=scoring_norm
        )

        self.sharding = sharding

        # self.entity_embedding[..., :embedding_size] : real part
        # self.entity_embedding[..., embedding_size:] : imaginary part
        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, 2 * embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, embedding_size
        )
        assert (
            self.entity_embedding.shape[-1]
            == 2 * self.relation_embedding.shape[-1]
            == 2 * embedding_size
        ), (
            "RotatE requires `2*embedding_size` embedding parameters for each entity"
            "and `embedding_size` embedding parameters for each relation"
        )
        self.embedding_size = embedding_size

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
        embedding_size: int,
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
            Size of entity and relation embeddings.
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
            self.entity_embedding.shape[-1]
            == self.relation_embedding.shape[-1]
            == embedding_size
        ), (
            "DistMult requires `embedding_size` embedding parameters"
            " for each entity and relation"
        )
        self.embedding_size = embedding_size

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
        embedding_size: int,
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
            Complex size of entity and relation embeddings.
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(ComplEx, self).__init__(negative_sample_sharing=negative_sample_sharing)

        self.sharding = sharding

        # self.entity_embedding[..., :embedding_size] : real part
        # self.entity_embedding[..., embedding_size:] : imaginary part
        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, 2 * embedding_size
        )
        # self.relation_embedding[..., :embedding_size] : real part
        # self.relation_embedding[..., embedding_size:] : imaginary part
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, 2 * embedding_size
        )
        assert (
            self.entity_embedding.shape[-1]
            == self.relation_embedding.shape[-1]
            == 2 * embedding_size
        ), (
            "ComplEx requires `2*embedding_size` embedding parameters"
            " for each entity and relation"
        )
        self.embedding_size = embedding_size

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


class BoxE(DistanceBasedScoreFunction):
    """
    BoxE scoring function :cite:p:`BoxE`.
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
        Initialize BoxE model.

        :param negative_sample_sharing:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type scoring_norm:
            see :meth:`DistanceBasedScoreFunction.__init__`
        :type sharding:
            Entity sharding.
        :param n_relation_type:
            Number of relation types in the knowledge graph.
        :param embedding_size:
            Size of final entity embeddings.
        :param entity_initializer:
            Initialization scheme or table for entity embeddings.
        :param relation_initializer:
            Initialization scheme or table for relation embeddings.
        """
        super(BoxE, self).__init__(
            negative_sample_sharing=negative_sample_sharing, scoring_norm=scoring_norm
        )

        self.sharding = sharding

        # self.entity_embedding[..., :embedding_size] : base positions
        # self.entity_embedding[..., embedding_size:] : translational bumps
        self.entity_embedding = initialize_entity_embedding(
            entity_initializer, self.sharding, 2 * embedding_size
        )
        # self.relation_embedding[..., :embedding_size] : head box centers
        # self.relation_embedding[..., embedding_size:2*embedding_size] : tail box centers
        # self.relation_embedding[..., 2*embedding_size:3*embedding_size] : head box widths
        # self.relation_embedding[..., 3*embedding_size:] : tail box widths
        self.relation_embedding = initialize_relation_embedding(
            relation_initializer, n_relation_type, 4 * embedding_size
        )
        assert (
            2 * self.entity_embedding.shape[-1]
            == self.relation_embedding.shape[-1]
            == 4 * embedding_size
        ), (
            "BoxE requires `2*embedding_size` embedding parameters for each entity"
            " and `4*embedding_size` embedding parameters for each relation"
        )
        self.embedding_size = embedding_size

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
        center_ht, width_ht = torch.split(relation_emb, 2 * self.embedding_size, dim=-1)
        width_ht = torch.abs(width_ht.view(-1, 2, self.embedding_size))
        width_plus1_ht = width_ht + torch.tensor(
            1.0, dtype=torch.float32, device=width_ht.device
        )
        k_ht = (
            torch.tensor(0.5, dtype=torch.float32, device=width_ht.device)
            * width_ht
            * (width_plus1_ht - torch.reciprocal(width_plus1_ht))
        )

        bumped_ht = (
            head_emb.view(-1, 2, self.embedding_size)
            + tail_emb.view(-1, 2, self.embedding_size)[:, [1, 0]]
        )
        center_dist_ht = torch.abs(
            bumped_ht - center_ht.view(-1, 2, self.embedding_size)
        )
        final_dist_ht = torch.where(
            torch.all(
                torch.le(center_dist_ht, torch.div(width_ht, 2.0)),
                dim=-1,
                keepdim=True,
            ),
            center_dist_ht / width_plus1_ht,
            center_dist_ht * width_plus1_ht - k_ht,
        )  # shape (batch_size, 2, emb_size)

        return -self.reduce_embedding(final_dist_ht).sum(-1)

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
        center_ht, width_ht = torch.split(relation_emb, 2 * self.embedding_size, dim=-1)
        width_ht = torch.abs(width_ht.view(-1, 1, 2, self.embedding_size))
        width_plus1_ht = width_ht + torch.tensor(
            1.0, dtype=torch.float32, device=width_ht.device
        )
        k_ht = (
            torch.tensor(0.5, dtype=torch.float32, device=width_ht.device)
            * width_ht
            * (width_plus1_ht - torch.reciprocal(width_plus1_ht))
        )

        if self.negative_sample_sharing:
            head_emb = head_emb.view(1, -1, 2 * self.embedding_size)

        bumped_ht = (
            head_emb.view(head_emb.shape[0], -1, 2, self.embedding_size)
            + tail_emb.view(-1, 1, 2, self.embedding_size)[:, :, [1, 0]]
        )

        center_dist_ht = torch.abs(
            bumped_ht - center_ht.view(-1, 1, 2, self.embedding_size)
        )
        final_dist_ht = torch.where(
            torch.all(
                torch.le(center_dist_ht, torch.div(width_ht, 2.0)),
                dim=-1,
                keepdim=True,
            ),
            center_dist_ht / width_plus1_ht,
            center_dist_ht * width_plus1_ht - k_ht,
        )  # shape (batch_size, B * n_heads, 2, emb_size)

        return -self.reduce_embedding(final_dist_ht).sum(-1)

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
        center_ht, width_ht = torch.split(relation_emb, 2 * self.embedding_size, dim=-1)
        width_ht = torch.abs(width_ht.view(-1, 1, 2, self.embedding_size))
        width_plus1_ht = width_ht + torch.tensor(
            1.0, dtype=torch.float32, device=width_ht.device
        )
        k_ht = (
            torch.tensor(0.5, dtype=torch.float32, device=width_ht.device)
            * width_ht
            * (width_plus1_ht - torch.reciprocal(width_plus1_ht))
        )

        if self.negative_sample_sharing:
            tail_emb = tail_emb.view(1, -1, 2 * self.embedding_size)

        bumped_ht = (
            head_emb.view(-1, 1, 2, self.embedding_size)
            + tail_emb.view(tail_emb.shape[0], -1, 2, self.embedding_size)[:, :, [1, 0]]
        )

        center_dist_ht = torch.abs(
            bumped_ht - center_ht.view(-1, 1, 2, self.embedding_size)
        )
        final_dist_ht = torch.where(
            torch.all(
                torch.le(center_dist_ht, torch.div(width_ht, 2.0)),
                dim=-1,
                keepdim=True,
            ),
            center_dist_ht / width_plus1_ht,
            center_dist_ht * width_plus1_ht - k_ht,
        )  # shape (batch_size, B * n_tails, 2, emb_size)

        return -self.reduce_embedding(final_dist_ht).sum(-1)
