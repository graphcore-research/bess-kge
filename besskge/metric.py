# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import torch


class BaseMetric(ABC):
    # Reduction function to apply along the batch dimension.
    reduction: Callable[[torch.Tensor], torch.Tensor]

    @abstractmethod
    def __call__(self, prediction_rank: torch.Tensor) -> torch.Tensor:
        """
        :param prediction_rank: shape: (batch_size,)
            Rank of ground truth among ordered predictions.

        :return:
            Reduced batch metric.
        """
        raise NotImplementedError


class ReciprocalRank(BaseMetric):
    """
    Reciprocal rank (e.g. to compute MRR).

    Returns (reduced) reciprocal rank of ground truth among predictions.
    """

    def __init__(self, reduction: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        :param reduction:
            see :class:`BaseMetric`
        """
        self.reduction = reduction

    # docstr-coverage: inherited
    def __call__(self, prediction_rank: torch.Tensor) -> torch.Tensor:
        return self.reduction(1.0 / prediction_rank)


class HitsAtK(BaseMetric):
    """
    Hits@K metric.

    Returns (reduced) count of triples where the ground truth
    is among the K most likely predicted entities.
    """

    def __init__(
        self, k: int, reduction: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """
        :param k:
            Maximum acceptable rank.
        :param reduction:
            see :class:`BaseMetric`
        """
        self.reduction = reduction
        self.K = k

    # docstr-coverage: inherited
    def __call__(self, prediction_rank: torch.Tensor) -> torch.Tensor:
        return self.reduction((prediction_rank <= self.K).to(torch.float))


# Metric str -> callable
METRICS_DICT = {"mrr": ReciprocalRank, "hits@k": HitsAtK}


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

        self.metrics: Dict[str, Callable[[torch.Tensor], torch.Tensor]]
        hits_k_filter = [re.search(r"hits@(\d+)", a) for a in metric_list]
        self.metrics = {
            a[0]: METRICS_DICT["hits@k"](k=int(a[1]), reduction=self.reduction)
            for a in hits_k_filter
            if a
        }
        self.metrics.update(
            {
                m_name: METRICS_DICT[m_name](reduction=self.reduction)
                for m_name in list(set(metric_list) - set(self.metrics.keys()))
            }
        )

    def metrics_from_scores(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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

        return {m_name: m_fn(batch_rank) for m_name, m_fn in self.metrics.items()}

    def metrics_from_indices(
        self, ground_truth: torch.Tensor, neg_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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

        return {m_name: m_fn(batch_rank) for m_name, m_fn in self.metrics.items()}
