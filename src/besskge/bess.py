from typing import List, Dict
import torch
import poptorch

from besskge.dataset import Sharding
from besskge.loss import BaseLossFunction
from besskge.scoring import BaseScoreFunction
from besskge.negative_sampler import RandomShardedNegativeSampler
from besskge.utils import all_to_all


class BessKGE(torch.nn.Module):
    def __init__(
        self,
        sharding: Sharding,
        n_relation_type: int,
        embedding_size: int,
        score_fn: BaseScoreFunction,
        loss_fn: BaseLossFunction,
        negative_sampler: RandomShardedNegativeSampler,
        initializer: str = "margin_based",
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.sharding = sharding
        self.n_relation_type = n_relation_type
        self.score_fn = score_fn
        self.loss_fn = loss_fn
        self.negative_sampler = negative_sampler
        if negative_sampler.flat_negative_format:
            assert (
                score_fn.negative_sample_sharing
            ), "Using flat negative format requires negative sample sharing"

        self.entity_embedding, self.relation_embedding = self.initialize_embeddings(
            initializer
        )

    def forward(
        self,
        head: torch.LongTensor,
        relation: torch.LongTensor,
        tail: torch.LongTensor,
        negative: torch.LongTensor,
        triple_weight: torch.FloatTensor,
    ) -> Dict[str, torch.FloatTensor]:
        head, relation, tail, negative, triple_weight = (
            head.squeeze(0),
            relation.squeeze(0),
            tail.squeeze(0),
            negative.squeeze(0),
            triple_weight.squeeze(0),
        )
        relation_embedding = self.relation_embedding[relation]
        negative_flat = negative.flatten(start_dim=1)
        gather_idx = torch.concat([head, tail, negative_flat], dim=1)
        head_embedding, tail_and_negative_embedding = torch.split(
            self.entity_embedding[gather_idx],
            [head.shape[1], tail.shape[1] + negative_flat.shape[1]],
            dim=1,
        )
        if self.negative_sampler.local_sampling:
            tail_embedding, negative_embedding = torch.split(
                tail_and_negative_embedding,
                [tail.shape[1], negative_flat.shape[1]],
                dim=1,
            )
            tail_embedding = all_to_all(tail_embedding)
        else:
            tail_and_negative_embedding = all_to_all(tail_and_negative_embedding)
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
                negative_heads = negative_tails = negative_embedding
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

        loss = self.loss_fn.compute_loss(positive_score, negative_score, triple_weight)
        return dict(
            loss=poptorch.identity_loss(loss, reduction="none"),
            positive_score=positive_score,
            negative_score=negative_score,
        )

    def initialize_embeddings(self, initializer: str) -> List[torch.nn.Parameter]:
        entity_embedding = torch.nn.Parameter(
            torch.FloatTensor(
                self.sharding.n_shard,
                self.sharding.max_entity_per_shard,
                self.embedding_size,
            )
        )
        # entity_embedding = torch.nn.Parameter(torch.FloatTensor(self.sharding.max_entity_per_shard, self.embedding_size))
        relation_embedding = torch.nn.Parameter(
            torch.FloatTensor(self.n_relation_type, self.embedding_size)
        )
        if initializer == "margin_based":
            embedding_range = self.loss_fn.margin / self.embedding_size
            torch.nn.init.uniform_(entity_embedding, -embedding_range, embedding_range)
            torch.nn.init.uniform_(
                relation_embedding, -embedding_range, embedding_range
            )
        else:
            raise ValueError(
                f"Initializer {initializer} not supported for embedding initialization"
            )
        return entity_embedding, relation_embedding
