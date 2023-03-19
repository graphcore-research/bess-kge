# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod

import poptorch
import torch


class BaseScoreFunction(ABC):
    """
    Base class for scoring functions.
    """

    def __init__(self, negative_sample_sharing: bool, *args, **kwargs) -> None:
        """
        Initialize scoring function.

        :param negative_sample_sharing:
            Share negative entities among all triples in the batch.
        """
        self.negative_sample_sharing = negative_sample_sharing

    @abstractmethod
    def score_triple(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Score (h,r,t) triples.

        :param head: shape: (batch_size, embedding_size)
            Embeddings of head entities in batch.
        :param relation: shape: (batch_size, embedding_size)
            Embeddings of relation types in batch.
        :param tail: shape: (batch_size, embedding_size)
            Embedding of tail entities in batch.

        :return: shape (batch_size,)
            Scores of batch triples.
        """
        raise NotImplementedError

    @abstractmethod
    def score_heads(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Score sets of head entities against fixed (r,t) queries.

        :param head: shape: (B, n_heads, embedding_size) with B = 1, batch_size
            Embeddings of head entities.
        :param relation: shape: (batch_size, embedding_size)
            Embeddings of relation types in batch.
        :param tail: shape: (batch_size, embedding_size)
            Embedding of tail entities in batch.

        :return: shape: (batch_size, B * n_heads) if
                :attr:`BaseScoreFunction.negative_sample_sharing`
                else (batch_size, n_heads)
            Scores of broadcasted triples.
        """
        raise NotImplementedError

    @abstractmethod
    def score_tails(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Score sets of tail entities against fixed (h,r) queries.

        :param head: shape: (batch_size, embedding_size)
            Embeddings of head entities in batch.
        :param relation: shape: (batch_size, embedding_size)
            Embeddings of relation types in batch.
        :param tail: shape: (B, n_tails, embedding_size) with B = 1, batch_size
            Embedding of tail entities.

        :return: shape: (batch_size, B * n_tails) if
                :attr:`BaseScoreFunction.negative_sample_sharing`
                else (batch_size, n_tails)
            Scores of broadcasted triples.
        """
        raise NotImplementedError


class DistanceBasedScoreFunction(BaseScoreFunction, ABC):
    """
    Base class for distance-based scoring functions.
    """

    def __init__(
        self, negative_sample_sharing: bool, scoring_norm: int, *args, **kwargs
    ) -> None:
        """
        Initialize distance-based scoring function.

        :param negative_sample_sharing:
            see :meth:`BaseScoreFunction.__init__`
        :param scoring_norm:
            p for p-norm to use in distance computation.
        """
        super(DistanceBasedScoreFunction, self).__init__(
            negative_sample_sharing, *args, **kwargs
        )
        self.scoring_norm = scoring_norm

    def reduce_norm(self, v: torch.FloatTensor) -> torch.FloatTensor:
        """
        p-norm reduction along embedding dimension.

        :param v: shape: (*, embedding_size)
            The tensor to reduce.

        :return: shape: (*,)
            p-norm reduction.
        """
        return torch.norm(v, p=self.scoring_norm, dim=-1)

    def distance_matrix(
        self, v1: torch.FloatTensor, v2: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Broadcasted pairwise distance between two collections of vectors.
        Computes p-norm reduction along trailing dimension of
        `tensor1[:, None, :] - tensor2[None, :, :]` without materializing the
        intermediate broadcasted difference, for memory optimization.

        :param v1: shape: (outer1, K)
            First collection.
        :param v2: shape: (outer2, K)
            Second collection.

        :return: shape: (outer1, outer2)
            Broadcasted pairwise p-distance.
        """
        if poptorch.isRunningOnIpu() and self.scoring_norm in [1, 2]:
            distance_matrix = poptorch.custom_op(
                name=f"L{self.scoring_norm}Distance",
                domain_version=1,
                domain="custom.ops",
                inputs=[v1, v2],
                example_outputs=[
                    torch.zeros(dtype=v1.dtype, size=[v1.shape[0], v2.shape[0]])
                ],
            )[0]
        else:
            distance_matrix = torch.cdist(v1, v2, p=self.scoring_norm)
        return distance_matrix

    def broadcasted_score(
        self, v1: torch.FloatTensor, v2: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Broadcasted scores of queries against sets of negative entities.

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
            dist = self.distance_matrix(v1, v2.reshape(-1, embedding_size))
        else:
            dist = self.reduce_norm(v1.unsqueeze(1) - v2)
        return dist


class TransE(DistanceBasedScoreFunction):
    """
    TransE scoring function (see [...]).
    """

    # docstr-coverage: inherited
    def score_triple(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -self.reduce_norm(head + relation - tail)

    # docstr-coverage: inherited
    def score_heads(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -self.broadcasted_score(tail - relation, head)

    # docstr-coverage: inherited
    def score_tails(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -self.broadcasted_score(head + relation, tail)
