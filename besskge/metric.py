import re
from typing import Dict, List

import torch


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
        hits_filter = [re.search(r"hits@(\d+)", a) for a in metric_list]
        self.hits_K_list = [int(a[1]) for a in hits_filter if a]
        self.compute_mrr = "mrr" in metric_list

    def compute_metrics(
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
        out_dict = {}

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

        for k in self.hits_K_list:
            out_dict.update(
                {f"hits@{k}": self.reduction((batch_rank <= k).to(torch.float))}
            )

        if self.compute_mrr:
            batch_mrr = 1.0 / batch_rank
            out_dict.update(mrr=self.reduction(batch_mrr))

        return out_dict
