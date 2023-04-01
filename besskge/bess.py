# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import poptorch
import torch
from poptorch_experimental_addons.collectives import (
    all_gather_cross_replica as all_gather,
)
from poptorch_experimental_addons.collectives import (
    all_to_all_single_cross_replica as all_to_all,
)

from besskge.embedding import (
    EmbeddingInitializer,
    initialize_entity_embedding,
    initialize_relation_embedding,
)
from besskge.loss import BaseLossFunction
from besskge.metric import Evaluation
from besskge.negative_sampler import (
    ShardedNegativeSampler,
    TripleBasedShardedNegativeSampler,
)
from besskge.scoring import BaseScoreFunction
from besskge.sharding import Sharding
from besskge.utils import gather_indices


class BessKGE(torch.nn.Module):
    """
    Base class for distributed training and inference of KGE models, using
    the distribution framework BESS [...].
    To be used in combination with a batch sampler based on a
    "ht_shardpair"-partitioned triple set.
    """

    def __init__(
        self,
        sharding: Sharding,
        n_relation_type: int,
        embedding_size: Optional[int],
        negative_sampler: ShardedNegativeSampler,
        entity_intializer: Union[torch.Tensor, EmbeddingInitializer],
        relation_intializer: Union[torch.Tensor, EmbeddingInitializer],
        score_fn: BaseScoreFunction,
        loss_fn: Optional[BaseLossFunction] = None,
        evaluation: Optional[Evaluation] = None,
        return_scores: bool = False,
    ) -> None:
        """
        Initialize BESS-KGE module.

        :param sharding:
            The entity sharding.
        :param n_relation_type:
            Number of relation types in the KG.
        :param embedding_size:
            Size of entities and relation embeddings.
        :param negative_sampler:
            Sampler of negative entities.
        :param entity_intializer:
            Initialization scheme / table for entity embeddings.
        :param relation_intializer:
            Initialization scheme / table for relation embeddings.
        :param score_fn:
            Scoring function.
        :param loss_fn:
            Loss function, required when training. Defaults to None.
        :param evaluation:
            Evaluation module, for computing metrics on device.
            Defaults to None.
        :param return_scores:
            Return positive and negative scores of batches to host.
            Defaults to False.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.sharding = sharding
        self.n_relation_type = n_relation_type
        self.negative_sampler = negative_sampler
        self.score_fn = score_fn
        self.loss_fn = loss_fn
        self.evaluation = evaluation
        self.return_scores = return_scores
        if not (loss_fn or evaluation or return_scores):
            raise ValueError(
                "Nothing to return. At least one between loss_fn,"
                " evaluation or return_scores needs to be != None"
            )

        if negative_sampler.flat_negative_format:
            assert (
                score_fn.negative_sample_sharing
            ), "Using flat negative format requires negative sample sharing"

        self.entity_embedding = initialize_entity_embedding(
            entity_intializer, self.sharding, self.embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_intializer, self.n_relation_type, self.embedding_size
        )

    def forward(
        self,
        head: torch.IntTensor,
        relation: torch.IntTensor,
        tail: torch.IntTensor,
        negative: torch.IntTensor,
        triple_weight: Optional[torch.FloatTensor] = None,
        negative_mask: Optional[torch.IntTensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward step, comprising of four phases:
        1) Gather relevant embeddings from local memory;
        2) Share embeddings with other devices through collective operators;
        3) Score positive and negative triples;
        4) Compute loss/metrics.
        Each device scores n_shard * positive_per_partition positive triples.

        :param head: shape: (n_shard, positive_per_partition)
            Head indices.
        :param relation: shape: (n_shard, positive_per_partition)
            Relation indices.
        :param tail: shape: (n_shard, positive_per_partition)
            Tail indices.
        :param negative: shape: (n_shard, B, n_negative)
            Indices of negative entities,
            with B = 1 or n_shard * positive_per_partition.
        :param triple_weight: shape: (n_shard * positive_per_partition,) or ()
            Weights of positive triples.

        :return:
            Microbatch loss, scores, metrics.
        """
        if triple_weight is None:
            triple_weight = torch.tensor(
                [1.0 / head.numel()], dtype=torch.float32, requires_grad=False
            )

        head, relation, tail, negative, triple_weight = (
            head.squeeze(0),
            relation.squeeze(0),
            tail.squeeze(0),
            negative.squeeze(0),
            triple_weight.squeeze(0),
        )

        positive_score, negative_score = self.score_batch(
            head, relation, tail, negative
        )

        if torch.is_tensor(negative_mask):
            negative_mask = negative_mask.squeeze(0)
            if (
                self.negative_sampler.flat_negative_format
                and self.negative_sampler.corruption_scheme == "ht"
            ):
                cutpoint = relation.shape[1] // 2
                mask_h, mask_t = torch.split(negative_mask, 1, dim=-2)
                negative_mask = torch.concat(
                    [
                        mask_h.expand(relation.shape[0], cutpoint, -1),
                        mask_t.expand(
                            relation.shape[0], relation.shape[1] - cutpoint, -1
                        ),
                    ],
                    dim=1,
                ).flatten(end_dim=1)
            negative_score = gather_indices(negative_score, negative_mask)

        out_dict = {}

        if self.return_scores:
            out_dict.update(
                positive_score=positive_score, negative_score=negative_score
            )

        if self.loss_fn:
            loss = self.loss_fn(
                positive_score,
                negative_score,
                triple_weight,
            )

            out_dict.update(
                loss=poptorch.identity_loss(loss, reduction="none"),
            )

        if self.evaluation:
            with torch.no_grad():
                out_dict.update(
                    metrics=self.evaluation.metrics_from_scores(
                        positive_score, negative_score
                    )
                )

        return out_dict

    @abstractmethod
    def score_batch(
        self,
        head: torch.IntTensor,
        relation: torch.IntTensor,
        tail: torch.IntTensor,
        negative: torch.IntTensor,
    ) -> Tuple[torch.Tensor]:
        """
        Compute positive and negative scores for the microbatch.

        :param head:
            see :meth:`BessKGE.forward`
        :param relation:
            see :meth:`BessKGE.forward`
        :param tail:
            see :meth:`BessKGE.forward`
        :param negative:
            see :meth:`BessKGE.forward`

        :return:
            positive (shape: (n_shard * positive_per_partition,))
            and negative (shape: (n_shard * positive_per_partition, n_negative))
            scores of the microbatch.
        """
        raise NotImplementedError


class EmbeddingMovingBessKGE(BessKGE):
    """
    Compute negative scores on the shard where the positive triples
    are scored (i.e. head shard).
    This requires moving embedding of negative entities between shards,
    which can be done with a single AllToAll collective.

    Each triple is scored against a total number of entities equal to
    `n_negative * n_shard` if negative sample sharing is disabled, otherwise
    `n_negative * n_shard * B` (see :meth:`BessKGE.forward`) for "h", "t"
    corruption scheme, `n_negative * n_shard * B // (B > 1 ? 2 : 1)` for "ht".
    """

    # docstr-coverage: inherited
    def score_batch(
        self,
        head: torch.IntTensor,
        relation: torch.IntTensor,
        tail: torch.IntTensor,
        negative: torch.IntTensor,
    ) -> Tuple[torch.Tensor]:
        # Gather embeddings
        n_shard = relation.shape[0]
        relation_embedding = self.relation_embedding[relation]
        negative_flat = negative.flatten(start_dim=1)
        gather_idx = torch.concat([head, tail, negative_flat], dim=1)
        head_embedding, tail_and_negative_embedding = torch.split(
            self.entity_embedding[gather_idx],
            [head.shape[1], tail.shape[1] + negative_flat.shape[1]],
            dim=1,
        )

        # Share negative and tail embeddings
        if self.negative_sampler.local_sampling:
            tail_embedding, negative_embedding = torch.split(
                tail_and_negative_embedding,
                [tail.shape[1], negative_flat.shape[1]],
                dim=1,
            )
            tail_embedding = all_to_all(tail_embedding, n_shard)
        else:
            tail_and_negative_embedding = all_to_all(
                tail_and_negative_embedding, n_shard
            )
            tail_embedding, negative_embedding = torch.split(
                tail_and_negative_embedding,
                [tail.shape[1], negative_flat.shape[1]],
                dim=1,
            )
        negative_embedding = (
            negative_embedding.reshape(*negative.shape, self.embedding_size)
            .transpose(0, 1)
            .flatten(start_dim=1, end_dim=2)
        )

        positive_score = self.score_fn.score_triple(
            head_embedding.flatten(end_dim=1),
            relation_embedding.flatten(end_dim=1),
            tail_embedding.flatten(end_dim=1),
        )

        if self.negative_sampler.corruption_scheme == "h":
            negative_score = self.score_fn.score_heads(
                negative_embedding,
                relation_embedding.flatten(end_dim=1),
                tail_embedding.flatten(end_dim=1),
            )
        elif self.negative_sampler.corruption_scheme == "t":
            negative_score = self.score_fn.score_tails(
                head_embedding.flatten(end_dim=1),
                relation_embedding.flatten(end_dim=1),
                negative_embedding,
            )
        elif self.negative_sampler.corruption_scheme == "ht":
            cut_point = relation.shape[1] // 2
            relation_half1, relation_half2 = torch.split(
                relation_embedding, cut_point, dim=1
            )
            if self.negative_sampler.flat_negative_format:
                negative_heads, negative_tails = torch.split(
                    negative_embedding, 1, dim=0
                )
            else:
                negative_embedding = negative_embedding.reshape(
                    *relation.shape[:2], -1, self.embedding_size
                )
                negative_heads, negative_tails = torch.split(
                    negative_embedding, cut_point, dim=1
                )
            negative_score_heads = self.score_fn.score_heads(
                negative_heads.flatten(end_dim=1),
                relation_half1.flatten(end_dim=1),
                tail_embedding[:, :cut_point, :].flatten(end_dim=1),
            )
            negative_score_tails = self.score_fn.score_tails(
                head_embedding[:, cut_point:, :].flatten(end_dim=1),
                relation_half2.flatten(end_dim=1),
                negative_tails.flatten(end_dim=1),
            )
            negative_score = torch.concat(
                [
                    negative_score_heads.reshape(*relation_half1.shape[:2], -1),
                    negative_score_tails.reshape(*relation_half2.shape[:2], -1),
                ],
                dim=1,
            ).flatten(end_dim=1)

        return positive_score, negative_score


class ScoreMovingBessKGE(BessKGE):
    """
    Compute negative scores on the shard where the negative entities are stored.
    This avoids moving embeddings between shards (convenient when the number of
    negative entities is very large, e.g. when scoring queries against all entities
    in the KG, or when using large embedding size).

    AllGather collectives are required to replicate queries on all devices, so that
    they can be scored against the local negative entities. An AllToAll collective
    is then used to send the scores back to the correct device.

    For the number of negative samples scored for each triple, see the corresponding
    value documented in :class:`EmbeddingMovingBessKGE` and, if using negative
    sample sharing, multiply that by n_shard.

    Does not support local sampling.
    """

    # docstr-coverage: inherited
    def score_batch(
        self,
        head: torch.IntTensor,
        relation: torch.IntTensor,
        tail: torch.IntTensor,
        negative: torch.IntTensor,
    ) -> Tuple[torch.Tensor]:
        n_shard = self.sharding.n_shard

        # Gather embeddings
        relation_embedding = self.relation_embedding[relation]
        negative_flat = negative.flatten(start_dim=1)
        gather_idx = torch.concat([head, tail, negative_flat], dim=1)
        head_embedding, tail_embedding, negative_embedding = torch.split(
            self.entity_embedding[gather_idx],
            [head.shape[1], tail.shape[1], negative_flat.shape[1]],
            dim=1,
        )
        negative_embedding = negative_embedding.reshape(
            *negative.shape, self.embedding_size
        )
        if (
            isinstance(self.negative_sampler, TripleBasedShardedNegativeSampler)
            and self.negative_sampler.flat_negative_format
        ):
            # Negatives are replicated along dimension 0, for local scoring only
            # one copy is needed
            negative_embedding = negative_embedding[0].unsqueeze(0)
        negative_embedding = negative_embedding.flatten(end_dim=1)

        relation_embedding_all = all_gather(relation_embedding, n_shard)

        if self.negative_sampler.corruption_scheme == "h":
            tail_embedding_all = all_gather(tail_embedding, n_shard).transpose(0, 1)
            negative_score = self.score_fn.score_heads(
                negative_embedding,
                relation_embedding_all.flatten(end_dim=2),
                tail_embedding_all.flatten(end_dim=2),
            )
        elif self.negative_sampler.corruption_scheme == "t":
            head_embedding_all = all_gather(head_embedding, n_shard)
            negative_score = self.score_fn.score_tails(
                head_embedding_all.flatten(end_dim=2),
                relation_embedding_all.flatten(end_dim=2),
                negative_embedding,
            )
        elif self.negative_sampler.corruption_scheme == "ht":
            cut_point = relation.shape[1] // 2
            relation_half1, relation_half2 = torch.split(
                relation_embedding_all, cut_point, dim=2
            )
            tail_embedding_all = all_gather(
                tail_embedding[:, :cut_point, :], n_shard
            ).transpose(0, 1)
            head_embedding_all = all_gather(head_embedding[:, cut_point:, :], n_shard)
            if self.negative_sampler.flat_negative_format:
                negative_heads, negative_tails = torch.split(
                    negative_embedding, 1, dim=0
                )
            else:
                negative_embedding = negative_embedding.reshape(
                    self.sharding.n_shard, *relation.shape[:2], -1, self.embedding_size
                )
                negative_heads, negative_tails = torch.split(
                    negative_embedding, cut_point, dim=2
                )
            negative_score_heads = self.score_fn.score_heads(
                negative_heads.flatten(end_dim=2),
                relation_half1.flatten(end_dim=2),
                tail_embedding_all.flatten(end_dim=2),
            )
            negative_score_tails = self.score_fn.score_tails(
                head_embedding_all.flatten(end_dim=2),
                relation_half2.flatten(end_dim=2),
                negative_tails.flatten(end_dim=2),
            )
            negative_score = torch.concat(
                [
                    negative_score_heads.reshape(*relation_half1.shape[:3], -1),
                    negative_score_tails.reshape(*relation_half2.shape[:3], -1),
                ],
                dim=2,
            ).flatten(end_dim=2)

        # Send negative scores back to corresponding triple processing device
        negative_score = (
            all_to_all(
                negative_score.reshape(
                    n_shard, relation.shape[0] * relation.shape[1], -1
                ),
                n_shard,
            )
            .transpose(0, 1)
            .flatten(start_dim=1)
        )

        # Recover microbatch tail embeddings (#TODO: avoidable?)
        tail_embedding = all_to_all(tail_embedding, n_shard)

        positive_score = self.score_fn.score_triple(
            head_embedding.flatten(end_dim=1),
            relation_embedding.flatten(end_dim=1),
            tail_embedding.flatten(end_dim=1),
        )

        return positive_score, negative_score
