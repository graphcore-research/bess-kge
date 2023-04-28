# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import torch


class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, prediction_rank: torch.Tensor) -> torch.Tensor:
        """
        :param prediction_rank: shape: (batch_size,)
            Rank of ground truth among ordered predictions.

        :return:
            Metric values for the element in the batch.
        """
        raise NotImplementedError


class ReciprocalRank(BaseMetric):
    """
    Reciprocal rank (e.g. to compute MRR).

    Returns reciprocal rank of ground truth among predictions.
    """

    def __init__(self) -> None:
        """
        :param reduction:
            see :class:`BaseMetric`
        """

    # docstr-coverage: inherited
    def __call__(self, prediction_rank: torch.Tensor) -> torch.Tensor:
        return torch.reciprocal(prediction_rank)


class HitsAtK(BaseMetric):
    """
    Hits@K metric.

    Returns count of triples where the ground truth
    is among the K most likely predicted entities.
    """

    def __init__(
        self,
        k: int,
    ) -> None:
        """
        :param k:
            Maximum acceptable rank.
        """
        self.K = k

    # docstr-coverage: inherited
    def __call__(self, prediction_rank: torch.Tensor) -> torch.Tensor:
        return (prediction_rank <= self.K).to(torch.float)


#: Metric str -> callable
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
        return_ranks: bool = False,
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
            Currently supports "none" (no reduction) and "sum".
        :param return_ranks:
            Whether to return prediction ranks alongside metrics.
        """
        if mode not in ["pessimistic", "optimistic", "average"]:
            raise ValueError(f"Mode {mode} not supported for evaluation")
        self.mode = mode
        self.return_ranks = return_ranks
        if reduction == "none":
            self.reduction = lambda x: x
        elif reduction == "sum":
            self.reduction = lambda x: torch.sum(x, dim=0)
        else:
            raise ValueError(f"Reduction {reduction} not supported for evaluation")
        self.worst_rank_infty = worst_rank_infty

        self.metrics: Dict[str, Callable[[torch.Tensor], torch.Tensor]]
        hits_k_filter = [re.search(r"hits@(\d+)", a) for a in metric_list]
        self.metrics = {
            a[0]: METRICS_DICT["hits@k"](k=int(a[1])) for a in hits_k_filter if a
        }
        self.metrics.update(
            {
                m_name: METRICS_DICT[m_name]()
                for m_name in list(set(metric_list) - set(self.metrics.keys()))
            }
        )

    def ranks_from_scores(
        self, pos_score: torch.Tensor, candidate_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the prediction rank from the score of the positive triple (ground truth) and
        the scores of triples completed with the candidate entities.

        :param pos_score: shape: (batch_size,)
            Scores of positive triples.
        :param candidate_score: shape: (batch_size, n_candidate)
            Scores of candidate triples.

        :return:
            The rank of the positive score among the ordered scores of the candidate triples.
        """
        batch_size, n_negative = candidate_score.shape
        pos_score = pos_score.reshape(-1, 1)
        if pos_score.shape[0] != batch_size:
            raise ValueError(
                "`pos_score` and `candidate_score` need to have same size at dimension 0"
            )

        if self.mode == "optimistic":
            n_better = torch.sum(candidate_score > pos_score, dim=-1).to(torch.float32)
            if self.worst_rank_infty:
                mask = n_better == n_negative
        elif self.mode == "pessimistic":
            n_better = torch.sum(candidate_score >= pos_score, dim=-1).to(torch.float32)
            if self.worst_rank_infty:
                mask = n_better == n_negative
        elif self.mode == "average":
            n_better_opt = torch.sum(candidate_score > pos_score, dim=-1).to(
                torch.float32
            )
            n_better_pess = torch.sum(candidate_score >= pos_score, dim=-1).to(
                torch.float32
            )
            n_better = torch.tensor(
                0.5, dtype=torch.float32, device=n_better_opt.device
            ) * (n_better_opt + n_better_pess)
            if self.worst_rank_infty:
                mask = torch.logical_or(
                    n_better_opt == n_negative, n_better_pess == n_negative
                )

        batch_rank = (
            torch.tensor(1.0, dtype=torch.float32, device=n_better.device) + n_better
        )
        if self.worst_rank_infty:
            batch_rank[mask] = torch.inf

        return batch_rank

    def ranks_from_indices(
        self, ground_truth: torch.Tensor, candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the prediction rank from the ground truth ID and
        ORDERED candidates' IDs.

        :param ground_truth: shape: (batch_size,)
            Indices of ground truth entities for each query.
        :param candidate_indices: shape: (batch_size, n_candidates)
            Indices of top n_candidates predicted entities,
            ordered by decreasing likelihood. The indices
            on each row are assumed to be distinct.

        :return:
            The rank of the ground truth among the predictions.
        """

        batch_size, n_negative = candidate_indices.shape
        ground_truth = ground_truth.reshape(-1, 1)
        if ground_truth.shape[0] != batch_size:
            raise ValueError(
                "`pos_score` and `candidate_score` need to have same size at dimension 0"
            )

        worst_rank = torch.inf if self.worst_rank_infty else float(n_negative + 1)
        ranks = torch.where(
            ground_truth == candidate_indices,
            torch.arange(
                1, n_negative + 1, dtype=torch.float32, device=ground_truth.device
            ),
            worst_rank,
        )
        batch_rank = ranks.min(dim=-1)[0]
        return batch_rank

    def dict_metrics_from_ranks(
        self, batch_rank: torch.Tensor, triple_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the required metrics starting from the prediction ranks
        of the elements in the batch.

        :param batch_rank: shape: (batch_size,)
            Prediction rank for each element in the batch.
        :param triple_mask: shape: (batch_size,)
            Boolean mask. If provided, all metrics for the elements where
            :code:`~triple_mask` are set to 0.0.

        :return:
            The dictionary of (reduced) batch metrics.
        """
        output = {}
        for m_name, m_fn in self.metrics.items():
            batch_metric = m_fn(batch_rank)
            if triple_mask is not None:
                batch_metric = torch.where(
                    triple_mask,
                    batch_metric,
                    torch.tensor(
                        [0.0], dtype=batch_metric.dtype, device=batch_metric.device
                    ),
                )
            output[m_name] = self.reduction(batch_metric)

        return output

    def stacked_metrics_from_ranks(
        self, batch_rank: torch.Tensor, triple_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Like :meth:`Evaluation.dict_metrics_from_ranks`, but the outputs for different
        metrics are returned stacked in a single tensor, according to the ordering
        of :attr:`Evaluation.metrics`.

        :param batch_rank: shape: (batch_size,)
            see :meth:`Evaluation.dict_metrics_from_ranks`.
        :param triple_mask: shape: (batch_size,)
            see :meth:`Evaluation.dict_metrics_from_ranks`.

        :return: shape: (1, n_metrics, batch_size)
            if :attr:`reduction` = "none", else (1, n_metrics)
            The stacked (reduced) metrics for the batch.
        """

        return torch.stack(
            list(self.dict_metrics_from_ranks(batch_rank, triple_mask).values())
        ).unsqueeze(0)
