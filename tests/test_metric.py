import torch

from besskge.metric import Evaluation
from torch.testing import assert_close


def test_evaluation() -> None:
    metric_list = ["mrr", "hits@1", "hits@5"]
    pos_score = torch.tensor([2.1, 5.0, 5.9])
    neg_score = torch.tensor(
        [
            [2.1, 3.1, 2.1, 5.2, 8.4],
            [9.8, 5.0, 1.0, 3.2, 5.0],
            [4.0, 2.3, 5.9, 3.1, 4.5],
        ]
    )

    evaluator_pess = Evaluation(metric_list, mode="pessimistic", reduction="none")
    results = evaluator_pess.compute_metrics(pos_score, neg_score)
    assert_close(results["hits@1"], torch.tensor([0.0, 0.0, 0.0]))
    assert_close(results["hits@5"], torch.tensor([0.0, 1.0, 1.0]))
    assert_close(results["mrr"], torch.tensor([1.0 / 6, 1.0 / 4, 1.0 / 2]))

    evaluator_opt = Evaluation(metric_list, mode="optimistic", reduction="none")
    results = evaluator_opt.compute_metrics(pos_score, neg_score)
    assert_close(results["hits@1"], torch.tensor([0.0, 0.0, 1.0]))
    assert_close(results["hits@5"], torch.tensor([1.0, 1.0, 1.0]))
    assert_close(results["mrr"], torch.tensor([1.0 / 4, 1.0 / 2, 1.0 / 1]))
