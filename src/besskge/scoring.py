from abc import ABC, abstractmethod
import torch
import poptorch


class BaseScoreFunction(ABC):
    def __init__(self, negative_sample_sharing: bool, *args, **kwargs):
        self.negative_sample_sharing = negative_sample_sharing

    @abstractmethod
    def score_triple(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def score_heads(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    @abstractmethod
    def score_tails(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class DistanceBasedScoreFunction(BaseScoreFunction, ABC):
    def __init__(
        self, negative_sample_sharing: bool, scoring_norm: int, *args, **kwargs
    ) -> None:
        super(DistanceBasedScoreFunction, self).__init__(
            negative_sample_sharing, *args, **kwargs
        )
        self.scoring_norm = scoring_norm

    def reduce_norm(self, v: torch.FloatTensor) -> torch.FloatTensor:
        """Norm reduction along the embedding dimension."""
        return torch.norm(v, p=self.scoring_norm, dim=-1)

    def cdist(self, v1: torch.FloatTensor, v2: torch.FloatTensor) -> torch.FloatTensor:
        if poptorch.isRunningOnIpu() and self.scoring_norm in [1, 2]:
            dist = poptorch.custom_op(
                name=f"L{self.scoring_norm}Distance",
                domain_version=1,
                domain="custom.ops",
                inputs=[v1, v2],
                example_outputs=[
                    torch.zeros(dtype=v1.dtype, size=[v1.shape[0], v2.shape[0]])
                ],
            )[0]
        else:
            dist = torch.cdist(v1, v2, p=self.scoring_norm)
        return dist

    def broadcasted_distance(
        self, v1: torch.FloatTensor, v2: torch.FloatTensor
    ) -> torch.FloatTensor:
        """v1 - float[batch_size, embedding_size]
        v2 - float[batch_size, n, embedding_size] or float[1, n, embedding_size]
        """
        embedding_size = v1.shape[-1]
        if self.negative_sample_sharing:
            dist = self.cdist(v1, v2.reshape(-1, embedding_size))
        else:
            dist = self.reduce_norm(v1.unsqueeze(-2) - v2)
        return dist


class TransE(DistanceBasedScoreFunction):
    def score_triple(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -self.reduce_norm(head + relation - tail)

    def score_heads(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -self.broadcasted_distance(tail - relation, head)

    def score_tails(
        self,
        head: torch.FloatTensor,
        relation: torch.FloatTensor,
        tail: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return -self.broadcasted_distance(head + relation, tail)
