import re
from functools import partial
from typing import Callable, Dict, List

import torch


def reciprocal_rank(
    prediction_rank: torch.tensor, reduction: Callable[[torch.tensor], torch.tensor]
) -> torch.tensor:
    """
    Reciprocal rank (e.g. to compute MRR).

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
        self,
        metric_list: List[str],
        mode: str = "average",
        worst_rank_infty: bool = False,
        reduction: str = "none",
    ) -> None:
        """
        Initialize evaluation module.

        :param metric_list:
            List of metrics to compute. Currently supports "mrr" and "hits@K".
        :param mode:
            "Optimistic"/"pessimistic"/"average" metrics from scores.
            Defaults to "average".
        :param worst_rank_infty:
            Assign a prediction rank of infinity as worst possible rank
            (as opposed to n_negative + 1). Defaults to False.
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
        self.worst_rank_infty = worst_rank_infty
        hits_k_filter = [re.search(r"hits@(\d+)", a) for a in metric_list]
        self.metrics = {
            a[0]: METRICS_DICT["hits@k"](int(a[1])) for a in hits_k_filter if a
        }
        self.metrics.update(
            {
                m_name: METRICS_DICT[m_name]
                for m_name in list(set(metric_list) - set(self.metrics.keys()))
            }
        )

    def metrics_from_scores(
        self, pos_score: torch.tensor, neg_score: torch.tensor
    ) -> Dict[str, torch.tensor]:
        """
        Compute the required metrics on a batch of scores.

        :param pos_score: shape: (batch_size,)
            Scores of positive triples.
        :param neg_score: shape: (batch_size, n_negative)
            Scores of negative triples.

        :return:
            The batch metrics.
        """

        batch_size, n_negative = neg_score.shape
        pos_score = pos_score.reshape(-1, 1)
        if pos_score.shape[0] != batch_size:
            raise ValueError(
                "`pos_score` and `neg_score` need to have same size at dimension 0"
            )

        if self.mode == "optimistic":
            n_better = torch.sum(neg_score > pos_score, dim=-1)
            if self.worst_rank_infty:
                mask = n_better == n_negative
        elif self.mode == "pessimistic":
            n_better = torch.sum(neg_score >= pos_score, dim=-1)
            if self.worst_rank_infty:
                mask = n_better == n_negative
        elif self.mode == "average":
            n_better_opt = torch.sum(neg_score > pos_score, dim=-1)
            n_better_pess = torch.sum(neg_score >= pos_score, dim=-1)
            n_better = (n_better_opt + n_better_pess) / 2
            if self.worst_rank_infty:
                mask = torch.logical_or(
                    n_better_opt == n_negative, n_better_pess == n_negative
                )

        batch_rank = 1.0 + n_better
        if self.worst_rank_infty:
            batch_rank[mask] = torch.inf

        return {
            m_name: m_fn(prediction_rank=batch_rank, reduction=self.reduction)
            for m_name, m_fn in self.metrics.items()
        }

    def metrics_from_indices(
        self, ground_truth: torch.tensor, neg_indices: torch.tensor
    ) -> Dict[str, torch.tensor]:
        """
        Compute the required metrics from the ground truth and
        ORDERED negative indices.

        :param ground_truth: shape: (batch_size,)
            Indices of ground truth entities for each query.
        :param neg_indices: shape: (batch_size, n_negative)
            Indices of top n_negative predicted entities,
            ordered by decreasing likelihood. The indices
            on each row are assumed to be distinct.

        :return:
            The batch metrics.
        """

        batch_size, n_negative = neg_indices.shape
        ground_truth = ground_truth.reshape(-1, 1)
        if ground_truth.shape[0] != batch_size:
            raise ValueError(
                "`pos_score` and `neg_score` need to have same size at dimension 0"
            )

        worst_rank = torch.inf if self.worst_rank_infty else float(n_negative + 1)
        batch_rank = torch.full((batch_size,), worst_rank, requires_grad=False)
        match_idx = torch.where(neg_indices == ground_truth)
        batch_rank[match_idx[0]] = 1.0 + match_idx[1]

        return {
            m_name: m_fn(prediction_rank=batch_rank, reduction=self.reduction)
            for m_name, m_fn in self.metrics.items()
        }
