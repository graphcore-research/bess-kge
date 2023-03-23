import re
from typing import Dict, List, Callable
from functools import partial

import torch


def reciprocal_rank(
    prediction_rank: torch.tensor, reduction: Callable[[torch.tensor], torch.tensor]
) -> torch.tensor:
    """
    Reciprocal rank (e.g. used to compute MRR)

    :param prediction_rank: shape: (batch_size,)
        Rank of ground truth among ordered predictions.
    :param reduction:
        Reduction function to apply along the batch dimension.
    :return:
        Reduced reciprocal rank.
    """
    return reduction(1.0 / prediction_rank)


def hits_at_k(
    prediction_rank: torch.tensor,
    k: int,
    reduction: Callable[[torch.tensor], torch.tensor],
) -> torch.tensor:
    """
    Hits@K metric.

    :param prediction_rank: shape: (batch_size,)
        Rank of ground truth among ordered predictions.
    :param k:
        Maximum acceptable rank.
    :param reduction:
        Reduction function to apply along the batch dimension.
    :return:
        Reduced count of predictions with rank at most k.
    """
    return reduction((prediction_rank <= k).to(torch.float))


METRICS_DICT = {"mrr": reciprocal_rank, "hits@k": (lambda k: partial(hits_at_k, k=k))}


class Evaluation:
    """
    A module for computing link prediction metrics.
    """

    def __init__(
        self, metric_list: List[str], mode: str = "average", reduction: str = "none"
    ) -> None:
        """
        Initialize evaluation module.

        :param metric_list:
            List of metrics to compute. Currently supports "mrr" and "hits@K".
        :param mode:
            Optimistic/pessimistic/average metrics, defaults to "average".
        :param reduction:
            How to reduce metrics along the batch dimension.
            Currently supports "none" (no reduction), "mean", "sum".
        """
        if mode not in ["pessimistic", "optimistic", "average"]:
            raise ValueError(f"Mode {mode} not supported for evaluation")
        self.mode = mode
        if reduction == "none":
            self.reduction = lambda x: x
        elif reduction == "sum":
            self.reduction = lambda x: torch.sum(x, dim=0)
        elif reduction == "mean":
            self.reduction = lambda x: torch.mean(x, dim=0)
        else:
            raise ValueError(f"Reduction {reduction} not supported for evaluation")
        hits_k_filter = [re.search(r"hits@(\d+)", a) for a in metric_list]
        self.metrics = {
            a[0]: METRICS_DICT["hits@k"](int(a[1])) for a in hits_k_filter if a
        }
        self.metrics.update(
            {
                m: METRICS_DICT[m]
                for m in list(set(metric_list) - set(self.metrics.keys()))
            }
        )

    def __call__(
        self, pos_score: torch.tensor, neg_score: torch.tensor
    ) -> Dict[str, torch.tensor]:
        """
        Compute the required metrics on a batch of scores.

        :param pos_score: shape: (batch_size,)
            Scores of positive triples.
        :param neg_score: shape: (batch_size, n_negatives)
            Scores of negative triples.

        :return:
            The batch metrics.
        """

        batch_size, _ = neg_score.shape
        pos_score = pos_score.reshape(-1, 1)
        if pos_score.shape[0] != batch_size:
            raise ValueError(
                "`pos_score` and `neg_score` need to have same size at dimension 0"
            )
        if self.mode == "optimistic":
            batch_rank = 1 + torch.sum(neg_score > pos_score, dim=-1)
        elif self.mode == "pessimistic":
            batch_rank = 1 + torch.sum(neg_score >= pos_score, dim=-1)
        elif self.mode == "average":
            rank_opt = 1 + torch.sum(neg_score > pos_score, dim=-1)
            rank_pess = 1 + torch.sum(neg_score >= pos_score, dim=-1)
            batch_rank = (rank_opt + rank_pess) / 2

        return {
            m_name: m(prediction_rank=batch_rank, reduction=self.reduction)
            for m_name, m in self.metrics.items()
        }
