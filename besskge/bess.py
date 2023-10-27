# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
PyTorch modules implementing the BESS distribution scheme :cite:p:`BESS`
for KGE training and inference on multiple IPUs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import poptorch
import torch
from poptorch_experimental_addons.collectives import (
    all_gather_cross_replica as all_gather,
)
from poptorch_experimental_addons.collectives import (
    all_to_all_single_cross_replica as all_to_all,
)

from besskge.loss import BaseLossFunction
from besskge.metric import Evaluation
from besskge.negative_sampler import (
    PlaceholderNegativeSampler,
    ShardedNegativeSampler,
    TripleBasedShardedNegativeSampler,
)
from besskge.scoring import BaseScoreFunction
from besskge.utils import gather_indices

BAD_NEGATIVE_SCORE = -50000.0


class BessKGE(torch.nn.Module, ABC):
    """
    Base class for distributed training and inference of KGE models, using
    the distribution framework BESS :cite:p:`BESS`.
    To be used in combination with a batch sampler based on a
    "ht_shardpair"-partitioned triple set.
    """

    def __init__(
        self,
        negative_sampler: ShardedNegativeSampler,
        score_fn: BaseScoreFunction,
        loss_fn: Optional[BaseLossFunction] = None,
        evaluation: Optional[Evaluation] = None,
        return_scores: bool = False,
        augment_negative: bool = False,
    ) -> None:
        """
        Initialize BESS-KGE module.

        :param negative_sampler:
            Sampler of negative entities.
        :param score_fn:
            Scoring function.
        :param loss_fn:
            Loss function, required when training. Default: None.
        :param evaluation:
            Evaluation module, for computing metrics on device.
            Default: None.
        :param return_scores:
            If True, return positive and negative scores of batches to the host.
            Default: False.
        :param augment_negative:
            If True, augment sampled negative entities with the head/tails
            (according to the corruption scheme) of other positive triples
            in the micro-batch. Default: False.
        """
        super().__init__()
        self.sharding = score_fn.sharding
        self.negative_sampler = negative_sampler
        self.score_fn = score_fn
        self.loss_fn = loss_fn
        self.evaluation = evaluation
        self.return_scores = return_scores
        self.augment_negative = augment_negative
        if not (loss_fn or evaluation or return_scores):
            raise ValueError(
                "Nothing to return. At least one of loss_fn,"
                " evaluation or return_scores needs to be != None"
            )

        if self.augment_negative:
            assert (
                score_fn.negative_sample_sharing
            ), "Negative augmentation requires negative sample sharing"
            assert not isinstance(
                self, ScoreMovingBessKGE
            ), "ScoreMovingBessKGE does not support negative augmentation"
        if negative_sampler.flat_negative_format:
            assert (
                score_fn.negative_sample_sharing
            ), "Using flat negative format requires negative sample sharing"
        elif score_fn.negative_sample_sharing and isinstance(
            self.negative_sampler, TripleBasedShardedNegativeSampler
        ):
            raise ValueError(
                "Negative sample sharing cannot be used"
                " with non-flat triple-specific negatives"
            )

        self.entity_embedding = self.score_fn.entity_embedding
        self.entity_embedding_size: int = self.score_fn.entity_embedding.shape[-1]

    @property
    def n_embedding_parameters(self) -> int:
        """
        Returns the number of trainable parameters in the embedding tables
        """
        return (
            self.score_fn.entity_embedding.numel()
            + self.score_fn.relation_embedding.numel()
        )

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
        triple_mask: Optional[torch.Tensor] = None,
        triple_weight: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        The forward step.

        Comprises of four phases:

        1) Gather relevant embeddings from local memory;

        2) Share embeddings with other devices through collective operators;

        3) Score positive and negative triples;

        4) Compute loss/metrics.

        Each device scores `n_shard * positive_per_partition` positive triples.

        :param head: shape: (1, n_shard, positive_per_partition)
            Head indices.
        :param relation: shape: (1, n_shard, positive_per_partition)
            Relation indices.
        :param tail: shape: (1, n_shard, positive_per_partition)
            Tail indices.
        :param triple_mask: shape: (1, n_shard, positive_per_partition)
            Mask to filter the triples in the micro-batch
            before computing metrics.
        :param negative: shape: (1, n_shard, B, padded_negative)
            Indices of negative entities,
            with `B = 1, 2 or n_shard * positive_per_partition`.
        :param triple_weight: shape: (1, n_shard * positive_per_partition,) or (1,)
            Weights of positive triples.
        :param negative_mask: shape: (1, B, n_shard, padded_negative)
            Mask to identify padding negatives, to discard when computing metrics.

        :return:
            Micro-batch loss, scores and metrics.
        """
        if triple_weight is None:
            triple_weight = torch.tensor(
                [1.0],
                dtype=torch.float32,
                requires_grad=False,
                device="ipu",
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

        if negative_mask is not None:
            negative_mask = negative_mask.squeeze(0).flatten(start_dim=-2)
            # shape (B, n_shard * padded_neg_length)
            if (
                self.negative_sampler.flat_negative_format
                and self.negative_sampler.corruption_scheme == "ht"
            ):
                cutpoint = relation.shape[1] // 2
                mask_h, mask_t = torch.split(negative_mask, 1, dim=0)
                negative_mask = torch.concat(
                    [
                        mask_h.expand(relation.shape[0], cutpoint, -1),
                        mask_t.expand(
                            relation.shape[0], relation.shape[1] - cutpoint, -1
                        ),
                    ],
                    dim=1,
                ).flatten(end_dim=1)

        if self.augment_negative:
            step = (
                1
                if self.negative_sampler.flat_negative_format
                else 1 + negative.shape[0] * negative.shape[2]
            )
            aug_mask = (
                torch.arange(
                    negative_score.shape[1],
                    device=negative_score.device,
                    dtype=torch.int,
                )[None, :]
                == step
                * torch.arange(
                    negative_score.shape[0],
                    device=negative_score.device,
                    dtype=torch.int,
                )[:, None]
            )
            if self.negative_sampler.corruption_scheme == "ht":
                aug_mask = (
                    aug_mask[: aug_mask.shape[0] // 2, :]
                    .reshape(relation.shape[0], relation.shape[1] // 2, -1)
                    .repeat(1, 2, 1)
                    .flatten(end_dim=1)
                )
            if negative_mask is not None:
                aug_mask[:, -negative_mask.shape[1] :] = ~negative_mask

            # Discard score of true head/tail from negative scores
            negative_score += (
                torch.tensor(
                    BAD_NEGATIVE_SCORE,
                    dtype=negative_score.dtype,
                    device=negative_score.device,
                )
                * aug_mask
            )
        elif negative_mask is not None:
            # Kill scores of padding negatives
            negative_score += torch.tensor(
                BAD_NEGATIVE_SCORE,
                dtype=negative_score.dtype,
                device=negative_score.device,
            ) * (~negative_mask)

        out_dict: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = dict()

        if self.return_scores:
            out_dict.update(
                positive_score=positive_score, negative_score=negative_score
            )

        if self.loss_fn:
            # Losses are always computed in FP32
            loss = self.loss_fn(
                positive_score.float(),
                negative_score.float(),
                triple_weight,
            )
            out_dict.update(loss=poptorch.identity_loss(loss, reduction="none"))

        if self.evaluation:
            if triple_mask is not None:
                triple_mask = triple_mask.flatten()
            with torch.no_grad():
                batch_rank = self.evaluation.ranks_from_scores(
                    positive_score, negative_score
                )
                if self.evaluation.return_ranks:
                    out_dict.update(ranks=batch_rank)
                out_dict.update(
                    metrics=self.evaluation.stacked_metrics_from_ranks(
                        batch_rank, triple_mask
                    )
                )

        return out_dict

    @abstractmethod
    def score_batch(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute positive and negative scores for the micro-batch.

        :param head:
            see :meth:`BessKGE.forward`
        :param relation:
            see :meth:`BessKGE.forward`
        :param tail:
            see :meth:`BessKGE.forward`
        :param negative:
            see :meth:`BessKGE.forward`

        :return:
            Positive (shape: (n_shard * positive_per_partition,))
            and negative (shape: (n_shard * positive_per_partition, n_negative))
            scores of the micro-batch.
        """
        raise NotImplementedError


class EmbeddingMovingBessKGE(BessKGE):
    """
    Compute negative scores on the shard where the positive triples
    are scored (namely the head shard).
    This requires moving the embedding of negative entities between shards,
    which can be done with a single AllToAll collective.

    Each triple is scored against a total number of entities equal to
    `n_negative * n_shard` if negative sample sharing is disabled, or to
    `n_negative * n_shard * B` otherwise (see :meth:`BessKGE.forward`) for "h", "t"
    corruption scheme, `n_negative * n_shard * (B > 2 ? B // 2 : 1)` for "ht".
    """

    # docstr-coverage: inherited
    def score_batch(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gather embeddings
        n_shard = relation.shape[0]
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
            negative_embedding.reshape(*negative.shape, self.entity_embedding_size)
            .transpose(0, 1)
            .flatten(start_dim=1, end_dim=2)
        )

        positive_score = self.score_fn.score_triple(
            head_embedding.flatten(end_dim=1),
            relation.flatten(end_dim=1),
            tail_embedding.flatten(end_dim=1),
        )

        if self.negative_sampler.corruption_scheme == "h":
            if self.augment_negative:
                negative_embedding = torch.concat(
                    [
                        head_embedding.view(
                            negative_embedding.shape[0], -1, self.entity_embedding_size
                        ),
                        negative_embedding,
                    ],
                    dim=1,
                )
            negative_score = self.score_fn.score_heads(
                negative_embedding,
                relation.flatten(end_dim=1),
                tail_embedding.flatten(end_dim=1),
            )
        elif self.negative_sampler.corruption_scheme == "t":
            if self.augment_negative:
                negative_embedding = torch.concat(
                    [
                        tail_embedding.view(
                            negative_embedding.shape[0], -1, self.entity_embedding_size
                        ),
                        negative_embedding,
                    ],
                    dim=1,
                )
            negative_score = self.score_fn.score_tails(
                head_embedding.flatten(end_dim=1),
                relation.flatten(end_dim=1),
                negative_embedding,
            )
        elif self.negative_sampler.corruption_scheme == "ht":
            cut_point = relation.shape[1] // 2
            relation_half1, relation_half2 = torch.split(
                relation,
                cut_point,
                dim=1,
            )
            head_half1, head_half2 = torch.split(
                head_embedding,
                cut_point,
                dim=1,
            )
            tail_half1, tail_half2 = torch.split(
                tail_embedding,
                cut_point,
                dim=1,
            )
            if self.negative_sampler.flat_negative_format:
                negative_heads, negative_tails = torch.split(
                    negative_embedding, 1, dim=0
                )
            else:
                negative_embedding = negative_embedding.reshape(
                    *relation.shape[:2], -1, self.entity_embedding_size
                )
                negative_heads, negative_tails = torch.split(
                    negative_embedding, cut_point, dim=1
                )
                negative_heads = negative_heads.flatten(end_dim=1)
                negative_tails = negative_tails.flatten(end_dim=1)
            if self.augment_negative:
                negative_heads = torch.concat(
                    [
                        head_half1.view(
                            negative_heads.shape[0], -1, self.entity_embedding_size
                        ),
                        negative_heads,
                    ],
                    dim=1,
                )
                negative_tails = torch.concat(
                    [
                        tail_half2.view(
                            negative_tails.shape[0], -1, self.entity_embedding_size
                        ),
                        negative_tails,
                    ],
                    dim=1,
                )

            negative_score_heads = self.score_fn.score_heads(
                negative_heads,
                relation_half1.flatten(end_dim=1),
                tail_half1.flatten(end_dim=1),
            )
            negative_score_tails = self.score_fn.score_tails(
                head_half2.flatten(end_dim=1),
                relation_half2.flatten(end_dim=1),
                negative_tails,
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
    negative entities is very large, for example when scoring queries against all entities
    in the knowledge graph, or when using a large embedding size).

    AllGather collectives are required to replicate queries on all devices, so that
    they can be scored against the local negative entities. An AllToAll collective
    is then used to send the scores back to the correct device.

    For the number of negative samples scored for each triple, see the corresponding
    value documented in :class:`EmbeddingMovingBessKGE` and, if using negative
    sample sharing, multiply that by `n_shard`.

    Does not support local sampling or negative augmentation.
    """

    # docstr-coverage: inherited
    def score_batch(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_shard = self.sharding.n_shard

        # Gather embeddings
        # relation_embedding = self.score_fn.relation_embedding[relation]
        negative_flat = negative.flatten(start_dim=1)
        gather_idx = torch.concat([head, tail, negative_flat], dim=1)
        head_embedding, tail_embedding, negative_embedding = torch.split(
            self.entity_embedding[gather_idx],
            [head.shape[1], tail.shape[1], negative_flat.shape[1]],
            dim=1,
        )
        negative_embedding = negative_embedding.reshape(
            *negative.shape, self.entity_embedding_size
        )
        if (
            isinstance(self.negative_sampler, TripleBasedShardedNegativeSampler)
            and self.negative_sampler.flat_negative_format
        ):
            # Negatives are replicated along dimension 0, for local scoring only
            # one copy is needed
            negative_embedding = negative_embedding[0].unsqueeze(0)

        relation_all = all_gather(relation, n_shard)

        if self.negative_sampler.corruption_scheme == "h":
            tail_embedding_all = all_gather(tail_embedding, n_shard).transpose(0, 1)
            negative_score = self.score_fn.score_heads(
                negative_embedding.flatten(end_dim=1),
                relation_all.flatten(end_dim=2),
                tail_embedding_all.flatten(end_dim=2),
            )
        elif self.negative_sampler.corruption_scheme == "t":
            head_embedding_all = all_gather(head_embedding, n_shard)
            negative_score = self.score_fn.score_tails(
                head_embedding_all.flatten(end_dim=2),
                relation_all.flatten(end_dim=2),
                negative_embedding.flatten(end_dim=1),
            )
        elif self.negative_sampler.corruption_scheme == "ht":
            cut_point = relation.shape[1] // 2
            relation_half1, relation_half2 = torch.split(
                relation_all,
                cut_point,
                dim=2,
            )
            tail_embedding_all = all_gather(
                tail_embedding[:, :cut_point, :], n_shard
            ).transpose(0, 1)
            head_embedding_all = all_gather(head_embedding[:, cut_point:, :], n_shard)
            if self.negative_sampler.flat_negative_format:
                negative_heads, negative_tails = torch.split(
                    negative_embedding, 1, dim=1
                )
                negative_heads = negative_heads.flatten(end_dim=1)
                negative_tails = negative_tails.flatten(end_dim=1)
            else:
                negative_embedding = negative_embedding.reshape(
                    self.sharding.n_shard,
                    *relation.shape[:2],
                    -1,
                    self.entity_embedding_size
                )
                negative_heads, negative_tails = torch.split(
                    negative_embedding, cut_point, dim=2
                )
                negative_heads = negative_heads.flatten(end_dim=2)
                negative_tails = negative_tails.flatten(end_dim=2)
            negative_score_heads = self.score_fn.score_heads(
                negative_heads,
                relation_half1.flatten(end_dim=2),
                tail_embedding_all.flatten(end_dim=2),
            )
            negative_score_tails = self.score_fn.score_tails(
                head_embedding_all.flatten(end_dim=2),
                relation_half2.flatten(end_dim=2),
                negative_tails,
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

        # Recover micro-batch tail embeddings (#TODO: avoidable?)
        tail_embedding = all_to_all(tail_embedding, n_shard)

        positive_score = self.score_fn.score_triple(
            head_embedding.flatten(end_dim=1),
            relation.flatten(end_dim=1),
            tail_embedding.flatten(end_dim=1),
        )

        return positive_score, negative_score


class TopKQueryBessKGE(torch.nn.Module):
    """
    Distributed scoring of (h, r, ?) or (?, r, t) queries (against
    all entities in the knowledge graph, or a query-specific set)
    returning the top-k most likely completions, based on the BESS :cite:p:`BESS`
    inference scheme.
    To be used in combination with a batch sampler based on a
    "h_shard"/"t_shard"-partitioned triple set.
    If the correct tail/head is known, this can be passed as an input
    in order to compute metrics on the final predictions.

    This class is recommended over :class:`BessKGE` when the number of
    negatives is large, for example when one wants to score queries against
    all entities in the knowledge graph, as it uses a sliding window over the
    negative sample size via an on-device for-loop.

    Only to be used for inference.
    """

    def __init__(
        self,
        k: int,
        candidate_sampler: Union[
            TripleBasedShardedNegativeSampler, PlaceholderNegativeSampler
        ],
        score_fn: BaseScoreFunction,
        evaluation: Optional[Evaluation] = None,
        return_scores: bool = False,
        window_size: int = 100,
    ) -> None:
        """
        Initialize TopK BESS-KGE module.

        :param k:
            For each query return the top-k most likely predictions.
        :param candidate_sampler:
            Sampler of candidate entities to score against queries.
            Use :class:`besskge.negative_sampler.PlaceholderNegativeSampler`
            to score queries against all entities in the knowledge graph, avoiding
            unnecessary loading of negative entities on device.
        :param score_fn:
            Scoring function.
        :param evaluation:
            Evaluation module, for computing metrics on device.
            Default: None.
        :param return_scores:
            If True, return scores of the top-k best completions.
            Default: False.
        :param window_size:
            Size of the sliding window, namely the number of negative entities
            scored against each query at each step of the on-device for-loop.
            Should be decreased with large batch sizes, to avoid an OOM error.
            Default: 100.
        """
        super().__init__()
        self.sharding = score_fn.sharding
        self.negative_sampler = candidate_sampler
        self.score_fn = score_fn
        self.evaluation = evaluation
        self.return_scores = return_scores
        self.k = k
        self.window_size = window_size

        if self.negative_sampler.flat_negative_format:
            assert (
                score_fn.negative_sample_sharing
            ), "Using flat negative format requires negative sample sharing"
        elif score_fn.negative_sample_sharing:
            raise ValueError(
                "Negative sample sharing cannot be used"
                " with non-flat triple-specific negatives"
            )

        if self.negative_sampler.corruption_scheme not in ["h", "t"]:
            raise ValueError("TopKQueryBessKGE only support 'h', 't' corruption scheme")

        if isinstance(self.negative_sampler, TripleBasedShardedNegativeSampler):
            assert self.negative_sampler.mask_on_gather, (
                "TopKQueryBessKGE requires setting mask_on_gather=True"
                " in the candidate_sampler"
            )

        self.entity_embedding = self.score_fn.entity_embedding
        self.entity_embedding_size: int = self.entity_embedding.shape[-1]

    def forward(
        self,
        relation: torch.Tensor,
        head: Optional[torch.Tensor] = None,
        tail: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
        triple_mask: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward step.

        Similarly to :class:`ScoreMovingBessKGE`, candidates are scored on the
        device where they are gathered, then scores for the same query against
        candidates in different shards are collected together via an AllToAll.
        At each iteration of the for loop, only the top-k best query responses
        and respective scores are kept to be used in the next iteration, while
        the rest are discarded.

        :param relation: shape: (1, shard_bs,)
            Relation indices.
        :param head: shape: (1, shard_bs,)
            Head indices, if known. Default: None.
        :param tail: shape: (1, shard_bs,)
            Tail indices, if known. Default: None.
        :param negative: shape: (1, n_shard, B, padded_negative)
            Candidates to score against the queries.
            It can be the same set for all queries (B=1),
            or specific for each query in the batch (B=shard_bs).
            If None, score each query against all entities in the knowledge
            graph. Default: None.
        :param triple_mask: shape: (1, shard_bs,)
            Mask to filter the triples in the micro-batch
            before computing metrics. Default: None.
        :param negative_mask: shape: (1, n_shard, B, padded_negative)
            If candidates are provided, mask to discard padding
            negatives when computing best completions.
            Requires the use of :code:`mask_on_gather=True` in the candidate
            sampler
            (see :class:`besskge.negative_sampler.TripleBasedShardedNegativeSampler`).
            Default: None.
        """

        relation = relation.squeeze(0)
        if head is not None:
            head = head.squeeze(0)
        if tail is not None:
            tail = tail.squeeze(0)

        candidate: torch.Tensor
        if negative is None:
            candidate = torch.arange(
                self.sharding.max_entity_per_shard,
                dtype=torch.int32,
                device=relation.device,
            )
        else:
            assert negative_mask is not None
            candidate = negative.squeeze(0)
            negative_mask = negative_mask.squeeze(0)
            if self.negative_sampler.flat_negative_format:
                candidate = candidate[0]
                negative_mask = negative_mask[0]
            negative_mask = negative_mask.reshape(-1, negative_mask.shape[-1])

        candidate = candidate.reshape(-1, candidate.shape[-1])
        # shape (1 or total_bs, n_negative_per_shard)

        n_shard = self.sharding.n_shard
        shard_bs = relation.shape[0]
        n_best = self.k + 1

        relation_all = all_gather(relation, n_shard)
        if self.negative_sampler.corruption_scheme == "h":
            tail_embedding = self.entity_embedding[tail]
            tail_embedding_all = all_gather(tail_embedding, n_shard)
        elif self.negative_sampler.corruption_scheme == "t":
            head_embedding = self.entity_embedding[head]
            head_embedding_all = all_gather(head_embedding, n_shard)

        def loop_body(
            curr_score: torch.Tensor, curr_idx: torch.Tensor, slide_idx: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mask = slide_idx < candidate.shape[-1]
            slide_idx = torch.where(
                mask,
                slide_idx,
                torch.tensor(
                    [candidate.shape[-1] - 1], dtype=torch.int32, device=mask.device
                ),
            )
            if negative_mask is not None:
                mask = torch.logical_and(mask, gather_indices(negative_mask, slide_idx))
            neg_ent_idx = gather_indices(
                candidate, slide_idx
            )  # shape (1 or n_sh * shard_bs, ws)
            negative_embedding = self.entity_embedding[neg_ent_idx]

            if self.negative_sampler.corruption_scheme == "h":
                negative_score = self.score_fn.score_heads(
                    negative_embedding,
                    relation_all.flatten(end_dim=1),
                    tail_embedding_all.flatten(end_dim=1),
                )
            elif self.negative_sampler.corruption_scheme == "t":
                negative_score = self.score_fn.score_tails(
                    head_embedding_all.flatten(end_dim=1),
                    relation_all.flatten(end_dim=1),
                    negative_embedding,
                )

            negative_score += torch.tensor(
                BAD_NEGATIVE_SCORE, dtype=negative_score.dtype, device=mask.device
            ) * (~mask).to(
                dtype=negative_score.dtype
            )  # shape (n_shard * shard_bs, ws)
            top_k_scores = torch.topk(
                torch.concat([negative_score, curr_score], dim=1),
                k=n_best,
                dim=1,
            )
            indices_broad = neg_ent_idx.broadcast_to(*negative_score.shape)
            indices = torch.concat([indices_broad, curr_idx], dim=1)
            curr_idx = gather_indices(indices, top_k_scores.indices)
            return (
                cast(torch.Tensor, top_k_scores.values),  # mypy check
                curr_idx,
                slide_idx
                + torch.tensor(
                    self.window_size, dtype=torch.int32, device=slide_idx.device
                ),
            )

        n_rep = int(np.ceil(candidate.shape[-1] / self.window_size))
        best_curr_score = torch.full(
            fill_value=BAD_NEGATIVE_SCORE,
            size=(n_shard * shard_bs, n_best),
            requires_grad=False,
            dtype=self.score_fn.entity_embedding.dtype,
            device=candidate.device,
        )
        best_curr_idx = torch.full(
            fill_value=self.sharding.max_entity_per_shard,
            size=(n_shard * shard_bs, n_best),
            requires_grad=False,
            dtype=torch.int32,
            device=candidate.device,
        )
        slide_idx = (
            torch.arange(self.window_size, dtype=torch.int32, device=relation.device)
            .to(torch.int32)
            .reshape(1, -1)
        )

        best_curr_score, best_curr_idx, _ = poptorch.for_loop(
            n_rep,
            loop_body,
            [
                best_curr_score,
                best_curr_idx,
                slide_idx,
            ],
        )  # shape (total_bs, n_best)

        # Send back queries to original shard
        best_score = all_to_all(
            best_curr_score.reshape(n_shard, shard_bs, n_best),
            n_shard,
        )
        best_idx = all_to_all(
            best_curr_idx.reshape(n_shard, shard_bs, n_best),
            n_shard,
        )

        # Discard padding shard entities
        best_score += torch.tensor(
            BAD_NEGATIVE_SCORE, dtype=best_score.dtype, device=best_idx.device
        ) * (
            best_idx
            >= torch.from_numpy(self.sharding.shard_counts)[:, None, None].to(
                dtype=torch.int32, device=best_idx.device
            )
        )

        # Best global indices
        best_global_idx = (
            gather_indices(
                torch.from_numpy(
                    self.sharding.shard_and_idx_to_entity,
                ).to(dtype=torch.int32, device=best_idx.device),
                best_idx.reshape(self.sharding.n_shard, -1),
            )
            .reshape(*best_idx.shape)
            .transpose(0, 1)
            .flatten(start_dim=1)
        )

        # Final topk among best k from all shards
        topk_final = torch.topk(
            best_score.transpose(0, 1).flatten(start_dim=1), k=self.k, dim=1
        )

        out_dict: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
        topk_global_id = gather_indices(best_global_idx, topk_final.indices)
        out_dict = dict(topk_global_id=topk_global_id)

        if self.return_scores:
            out_dict.update(topk_scores=topk_final.values)

        if self.evaluation:
            if triple_mask is not None:
                triple_mask = triple_mask.flatten()
            with torch.no_grad():
                ground_truth = (
                    tail if self.negative_sampler.corruption_scheme == "t" else head
                )
                assert (
                    ground_truth is not None
                ), "Evaluation requires providing ground truth entities"
                batch_rank = self.evaluation.ranks_from_indices(
                    ground_truth, topk_global_id
                )
                if self.evaluation.return_ranks:
                    out_dict.update(ranks=batch_rank)
                out_dict.update(
                    metrics=self.evaluation.stacked_metrics_from_ranks(
                        batch_rank, triple_mask
                    )
                )

        return out_dict


class AllScoresBESS(torch.nn.Module):
    """
    Distributed scoring of (h, r, ?) or (?, r, t) queries against
    the entities in the knowledge graph, returning all scores to
    host in blocks, based on the BESS :cite:p:`BESS`
    inference scheme.
    To be used in combination with a batch sampler based on a
    "h_shard"/"t_shard"-partitioned triple set.
    Since each iteration on IPU computes only part of the scores
    (based on the size of the sliding window), metrics should be
    computed on host after aggregating data (see
    :class:`besskge.pipeline.AllScoresPipeline`).

    Only to be used for inference.
    """

    def __init__(
        self,
        candidate_sampler: PlaceholderNegativeSampler,
        score_fn: BaseScoreFunction,
        window_size: int = 1000,
    ) -> None:
        """
        Initialize AllScores BESS-KGE module.

        :param candidate_sampler:
            :class:`besskge.negative_sampler.PlaceholderNegativeSampler` class,
            specifying corruption scheme.
        :param score_fn:
            Scoring function.
        :param window_size:
            Size of the sliding window, namely the number of negative entities
            scored against each query at each step on IPU and returned to host.
            Should be decreased with large batch sizes, to avoid an OOM error.
            Default: 1000.
        """
        super().__init__()
        self.sharding = score_fn.sharding
        self.score_fn = score_fn
        self.negative_sampler = candidate_sampler
        self.window_size = window_size

        if not score_fn.negative_sample_sharing:
            raise ValueError("AllScoresBESS requires using negative sample sharing")

        if self.negative_sampler.corruption_scheme not in ["h", "t"]:
            raise ValueError("AllScoresBESS only support 'h', 't' corruption scheme")

        if not isinstance(self.negative_sampler, PlaceholderNegativeSampler):
            raise ValueError(
                "AllScoresBESS requires a `PlaceholderNegativeSampler`"
                " candidate_sampler"
            )

        self.entity_embedding = self.score_fn.entity_embedding
        self.entity_embedding_size: int = self.entity_embedding.shape[-1]

        self.candidate = torch.arange(self.window_size, dtype=torch.int32)
        self.n_step = int(
            np.ceil(self.sharding.max_entity_per_shard / self.window_size)
        )

    def forward(
        self,
        step: torch.Tensor,
        relation: torch.Tensor,
        head: Optional[torch.Tensor] = None,
        tail: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward step.

        Similarly to :class:`ScoreMovingBessKGE`, candidates are scored on the
        device where they are gathered, then scores for the same query against
        candidates in different shards are collected together via an AllToAll.

        :param step:
            The index of the block (of size self.window_size) of entities
            on each IPU to score against queries.
        :param relation: shape: (1, shard_bs,)
            Relation indices.
        :param head: shape: (1, shard_bs,)
            Head indices, if known. Default: None.
        :param tail: shape: (1, shard_bs,)
            Tail indices, if known. Default: None.

        :return:
            The scores for the completions.
        """

        relation = relation.squeeze(0)
        if head is not None:
            head = head.squeeze(0)
        if tail is not None:
            tail = tail.squeeze(0)

        n_shard = self.sharding.n_shard
        shard_bs = relation.shape[0]

        relation_all = all_gather(relation, n_shard)
        if self.negative_sampler.corruption_scheme == "h":
            tail_embedding = self.entity_embedding[tail]
            tail_embedding_all = all_gather(tail_embedding, n_shard)
        elif self.negative_sampler.corruption_scheme == "t":
            head_embedding = self.entity_embedding[head]
            head_embedding_all = all_gather(head_embedding, n_shard)

        # Local indices of the entities to score against queries
        ent_slice = torch.minimum(
            step * self.window_size
            + torch.arange(self.window_size, device=relation.device),
            torch.tensor(self.sharding.max_entity_per_shard - 1),
        )
        negative_embedding = self.entity_embedding[ent_slice]

        if self.negative_sampler.corruption_scheme == "h":
            scores = self.score_fn.score_heads(
                negative_embedding,
                relation_all.flatten(end_dim=1),
                tail_embedding_all.flatten(end_dim=1),
            )
        elif self.negative_sampler.corruption_scheme == "t":
            scores = self.score_fn.score_tails(
                head_embedding_all.flatten(end_dim=1),
                relation_all.flatten(end_dim=1),
                negative_embedding,
            )

        # Send back queries to original shard
        scores = (
            all_to_all(
                scores.reshape(n_shard, shard_bs, self.window_size),
                n_shard,
            )
            .transpose(0, 1)
            .flatten(start_dim=1)
        )  # shape (bs, n_shard * ws)

        return scores
