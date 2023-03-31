import torch
import pytest

from besskge.metric import Evaluation
from torch.testing import assert_close

metric_list = ["mrr", "hits@1", "hits@5"]


@pytest.mark.parametrize("worst_rank_infty", [True, False])
def test_metrics_from_scores(worst_rank_infty: bool) -> None:
    pos_score = torch.tensor([2.1, 5.0, 5.9, 2.0])
    neg_score = torch.tensor(
        [
            [2.1, 3.1, 2.1, 5.2, 8.4],
            [9.8, 5.0, 1.0, 3.2, 5.0],
            [4.0, 2.3, 5.9, 3.1, 4.5],
            [4.0, 2.3, 5.9, 3.1, 4.5],
        ]
    )

    worst_rr = 0.0 if worst_rank_infty else 1.0 / 6

    evaluator_pess = Evaluation(
        metric_list,
        mode="pessimistic",
        reduction="none",
        worst_rank_infty=worst_rank_infty,
    )

    results = evaluator_pess.metrics_from_scores(pos_score, neg_score)
    assert_close(results["hits@1"], torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert_close(results["hits@5"], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert_close(results["mrr"], torch.tensor([worst_rr, 1.0 / 4, 1.0 / 2, worst_rr]))

    evaluator_opt = Evaluation(
        metric_list,
        mode="optimistic",
        reduction="none",
        worst_rank_infty=worst_rank_infty,
    )
    results = evaluator_opt.metrics_from_scores(pos_score, neg_score)
    assert_close(results["hits@1"], torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert_close(results["hits@5"], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert_close(results["mrr"], torch.tensor([1.0 / 4, 1.0 / 2, 1.0 / 1, worst_rr]))


@pytest.mark.parametrize("worst_rank_infty", [True, False])
def test_metrics_from_indices(worst_rank_infty: bool) -> None:
    ground_truth = torch.tensor([6, 0, 2])
    neg_indices = torch.tensor(
        [
            [6, 1, 45, 33, 28],
            [5, 2, 12, 0, 44],
            [27, 9, 1, 6, 17],
        ]
    )

    worst_rr = 0.0 if worst_rank_infty else 1.0 / 6

    evaluator_pess = Evaluation(
        metric_list,
        reduction="none",
        worst_rank_infty=worst_rank_infty,
    )

    results = evaluator_pess.metrics_from_indices(ground_truth, neg_indices)
    assert_close(results["hits@1"], torch.tensor([1.0, 0.0, 0.0]))
    assert_close(results["hits@5"], torch.tensor([1.0, 1.0, 0.0]))
    assert_close(results["mrr"], torch.tensor([1.0, 1.0 / 4, worst_rr]))
