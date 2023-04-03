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
            entity_intializer, self.sharding, embedding_size
        )
        self.relation_embedding = initialize_relation_embedding(
            relation_intializer, self.n_relation_type, embedding_size
        )
        self.embedding_size: int = self.entity_embedding.shape[-1]

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
        triple_weight: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
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
                [1.0 / head.numel()],
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

        out_dict: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = dict()

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
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        relation_embedding_all = all_gather(relation_embedding, n_shard)

        if self.negative_sampler.corruption_scheme == "h":
            tail_embedding_all = all_gather(tail_embedding, n_shard).transpose(0, 1)
            negative_score = self.score_fn.score_heads(
                negative_embedding.flatten(end_dim=1),
                relation_embedding_all.flatten(end_dim=2),
                tail_embedding_all.flatten(end_dim=2),
            )
        elif self.negative_sampler.corruption_scheme == "t":
            head_embedding_all = all_gather(head_embedding, n_shard)
            negative_score = self.score_fn.score_tails(
                head_embedding_all.flatten(end_dim=2),
                relation_embedding_all.flatten(end_dim=2),
                negative_embedding.flatten(end_dim=1),
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
                    negative_embedding, 1, dim=1
                )
                negative_heads = negative_heads.flatten(end_dim=1)
                negative_tails = negative_tails.flatten(end_dim=1)
            else:
                negative_embedding = negative_embedding.reshape(
                    self.sharding.n_shard, *relation.shape[:2], -1, self.embedding_size
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

        # Recover microbatch tail embeddings (#TODO: avoidable?)
        tail_embedding = all_to_all(tail_embedding, n_shard)

        positive_score = self.score_fn.score_triple(
            head_embedding.flatten(end_dim=1),
            relation_embedding.flatten(end_dim=1),
            tail_embedding.flatten(end_dim=1),
        )

        return positive_score, negative_score


# class TopKQueryBessKGE(torch.nn.Module):
#     """
#     Distributed scoring of (h, r, ?) or (?, r, t) queries (against
#     all entities in the KG, or a query-specific set)
#     returning top-k most likely completions, based on BESS
#     inference scheme.
#     To be used in combination with a batch sampler based on a
#     "h_shard"/"t_shard"-partitioned triple set.
#     """

#     def forward(
#         self,
#         head: Optional[torch.Tensor],
#         relation: torch.Tensor,
#         tail: Optional[torch.Tensor],
#         negative: Optional[torch.Tensor] = None,
#     ) -> Dict[str, Any]:
#         """
#         _summary_

#         :param head: _description_
#         :type head: _type_
#         :param relation: _description_
#         :type relation: _type_
#         :param tail: _description_
#         :type tail: _type_
#         :param negative: _description_, defaults to None
#         :type negative: _type_, optional
#         :return: _description_
#         :rtype: _type_
#         """
#         head, relation, tail, negative = (
#             head.squeeze(0),
#             relation.squeeze(0),
#             tail.squeeze(0),
#             negative.squeeze(0),
#         )
#         n_shard = self.sharding.n_shard

#         # Gather embeddings
#         relation_embedding = self.relation_embedding[relation]
#         negative_flat = negative.flatten(start_dim=1)
#         gather_idx = torch.concat([head, tail, negative_flat], dim=1)
#         head_embedding, tail_embedding, negative_embedding = torch.split(
#             self.entity_embedding[gather_idx],
#             [head.shape[1], tail.shape[1], negative_flat.shape[1]],
#             dim=1,
#         )
#         negative_embedding = negative_embedding.reshape(
#             *negative.shape, self.embedding_size
#         ).flatten(end_dim=1)

#         relation_embedding_all = all_gather(relation_embedding, n_shard)

#         def loop_body(curr_score, curr_idx, neg_idx):
#             mask = neg_idx >= negative.shape[-1]
#             gath_idx = negative[..., neg_idx].flatten(end_dim=1)  # shape (n_sh * B, ws)
#             # mask = ent_idx >= sharding.max_entity_per_shard
#             # gath_idx = neg_idx.reshape(1,-1) # shape (1, ws)
#             negative_embedding = self.entity_embedding[gath_idx]

#             if self.negative_sampler.corruption_scheme == "h":
#                 tail_embedding_all = all_gather(tail_embedding, n_shard).transpose(0, 1)
#                 negative_score = self.score_fn.score_heads(
#                     negative_embedding,
#                     relation_embedding_all.flatten(end_dim=2),
#                     tail_embedding_all.flatten(end_dim=2),
#                 )
#             elif self.negative_sampler.corruption_scheme == "t":
#                 head_embedding_all = all_gather(head_embedding, n_shard)
#                 negative_score = self.score_fn.score_tails(
#                     head_embedding_all.flatten(end_dim=2),
#                     relation_embedding_all.flatten(end_dim=2),
#                     negative_embedding,
#                 )
#             elif self.negative_sampler.corruption_scheme == "ht":
#                 cut_point = relation.shape[1] // 2
#                 relation_half1, relation_half2 = torch.split(
#                     relation_embedding_all, cut_point, dim=2
#                 )
#                 tail_embedding_all = all_gather(
#                     tail_embedding[:, :cut_point, :], n_shard
#                 ).transpose(0, 1)
#                 head_embedding_all = all_gather(
#                     head_embedding[:, cut_point:, :], n_shard
#                 )
#                 if self.negative_sampler.flat_negative_format:
#                     negative_heads, negative_tails = torch.split(
#                         negative_embedding, 1, dim=0
#                     )
#                 else:
#                     negative_embedding = negative_embedding.reshape(
#                         self.sharding.n_shard,
#                         *relation.shape[:2],
#                         -1,
#                         self.embedding_size,
#                     )
#                     negative_heads, negative_tails = torch.split(
#                         negative_embedding, cut_point, dim=2
#                     )
#                 negative_score_heads = self.score_fn.score_heads(
#                     negative_heads.flatten(end_dim=2),
#                     relation_half1.flatten(end_dim=2),
#                     tail_embedding_all.flatten(end_dim=2),
#                 )
#                 negative_score_tails = self.score_fn.score_tails(
#                     head_embedding_all.flatten(end_dim=2),
#                     relation_half2.flatten(end_dim=2),
#                     negative_tails.flatten(end_dim=2),
#                 )
#                 negative_score = torch.concat(
#                     [
#                         negative_score_heads.reshape(*relation_half1.shape[:3], -1),
#                         negative_score_tails.reshape(*relation_half2.shape[:3], -1),
#                     ],
#                     dim=2,
#                 ).flatten(end_dim=2)

#             # negative_score ha ora shape (total_bs, ws)
#             negative_score = negative_score - 10000 * mask
#             top_k_scores = torch.topk(
#                 torch.concat([negative_score, curr_score], dim=1), k=self.n_best, dim=1
#             )
#             curr_score = top_k_scores.values
#             indices_broad = neg_idx.reshape(1, -1).broadcast_to(*negative_score.shape)
#             indices = torch.concat([indices_broad, curr_idx], dim=1)
#             curr_idx = gather_indices(indices, top_k_scores.indices)
#             return curr_score, curr_idx, neg_idx + self.window_size

#         n_rep = int(np.ceil(negative.shape[-1] / self.window_size))
#         # n_rep = int(np.ceil(self.sharding.max_entity_per_shard / self.window_size))
#         best_curr_score = -10000.0 * torch.ones(
#             size=(n_shard * relation.shape[0] * relation.shape[1], self.n_best),
#             requires_grad=False,
#         )
#         best_curr_idx = self.sharding.max_entity_per_shard * torch.ones(
#             size=(n_shard * relation.shape[0] * relation.shape[1], self.n_best),
#             requires_grad=False,
#         ).to(torch.int32)
#         best_curr_score, best_curr_idx, _ = poptorch.for_loop(
#             n_rep,
#             loop_body,
#             [
#                 best_curr_score,
#                 best_curr_idx,
#                 torch.arange(self.window_size).to(torch.int32),
#             ],
#         )  # shape (total_bs, n_best)

#         # Send back queries to original shard
#         best_score = all_to_all(
#             best_curr_score.reshape(
#                 n_shard, relation.shape[0] * relation.shape[1], self.n_best
#             ),
#             n_shard,
#         )
#         best_idx = all_to_all(
#             best_curr_idx.reshape(
#                 n_shard, relation.shape[0] * relation.shape[1], self.n_best
#             ),
#             n_shard,
#         )

# INDICI GLOBALI?

# # Reconstruct global indices
# global_best_idx = gather_indices(torch.from_numpy(sharding.shard_and_idx_to_entity).to(device=best_idx.device), best_idx.reshape(self.sharding.n_shard, -1)).reshape(*best_idx.shape)
# global_best_idx = global_best_idx.transpose(0,1).reshape(self.shard_bs, -1)

# #Final topk among best k from all shards
# top_k_final = torch.topk(best_score.transpose(0,1).reshape(self.shard_bs, -1), k=self.n_best, dim=1)
# final_best_idx = gather_indices(global_best_idx, top_k_final.indices)

# --------------------------------

# # Send negative scores back to corresponding triple processing device
# negative_score = (
#     all_to_all(
#         negative_score.reshape(
#             n_shard, relation.shape[0] * relation.shape[1], -1
#         )
#     )
#     .transpose(0, 1)
#     .flatten(start_dim=1)
# )

# # Recover microbatch tail embeddings (#TODO: avoidable?)
# tail_embedding = all_to_all(tail_embedding)

# positive_score = self.score_fn.score_triple(
#     head_embedding.flatten(end_dim=1),
#     relation_embedding.flatten(end_dim=1),
#     tail_embedding.flatten(end_dim=1),
# )

# loss = self.loss_fn(
#     positive_score,
#     negative_score,
#     triple_weight,
# )

# out_dict = dict(
#     loss=poptorch.identity_loss(loss, reduction="none"),
# )

# if self.return_scores:
#     out_dict.update(
#         positive_score=positive_score, negative_score=negative_score
#     )

# return out_dict
