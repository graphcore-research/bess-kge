# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from besskge.batch_sampler import RigidShardedBatchSampler
from besskge.dataset import KGDataset
from besskge.metric import Evaluation
from besskge.negative_sampler import PlaceholderNegativeSampler
from besskge.pipeline import AllScoresPipeline
from besskge.scoring import ComplEx
from besskge.sharding import PartitionedTripleSet, Sharding
from besskge.utils import get_entity_filter

seed = 1234
n_entity = 5000
n_relation_type = 50
n_shard = 4
n_test_triple = 1000
batches_per_step = 2
shard_bs = 400
embedding_size = 128

np.random.seed(seed)
torch.manual_seed(seed)

sharding = Sharding.create(n_entity, n_shard, seed=seed)

unsharded_entity_table = torch.randn(size=(n_entity, 2 * embedding_size))
relation_table = torch.randn(size=(n_relation_type, 2 * embedding_size))

test_triples_h = np.random.choice(n_entity - 1, size=n_test_triple, replace=False)
test_triples_t = np.random.choice(n_entity - 1, size=n_test_triple, replace=False)
test_triples_r = np.random.randint(n_relation_type, size=n_test_triple)
triples = {"test": np.stack([test_triples_h, test_triples_r, test_triples_t], axis=1)}


@pytest.mark.parametrize("corruption_scheme", ["h", "t"])
@pytest.mark.parametrize(
    "filter_scores, extra_only", [(True, True), (True, False), (False, False)]
)
def test_all_scores_pipeline(
    corruption_scheme: str, filter_scores: bool, extra_only: bool
) -> None:
    ds = KGDataset(
        n_entity=n_entity,
        n_relation_type=n_relation_type,
        entity_dict=None,
        relation_dict=None,
        type_offsets=None,
        triples=triples,
        original_triple_ids={k: np.arange(v.shape[0]) for k, v in triples.items()},
    )

    partition_mode = "h_shard" if corruption_scheme == "t" else "t_shard"
    partitioned_triple_set = PartitionedTripleSet.create_from_dataset(
        ds, "test", sharding, partition_mode=partition_mode
    )

    score_fn = ComplEx(
        negative_sample_sharing=True,
        sharding=sharding,
        n_relation_type=ds.n_relation_type,
        embedding_size=embedding_size,
        entity_initializer=unsharded_entity_table,
        relation_initializer=relation_table,
    )
    placeholder_ns = PlaceholderNegativeSampler(
        corruption_scheme=corruption_scheme, seed=seed
    )

    test_bs = RigidShardedBatchSampler(
        partitioned_triple_set=partitioned_triple_set,
        negative_sampler=placeholder_ns,
        shard_bs=shard_bs,
        batches_per_step=batches_per_step,
        seed=seed,
        hrt_freq_weighting=False,
        duplicate_batch=False,
        return_triple_idx=True,
    )

    evaluation = Evaluation(
        ["mrr", "hits@10"], mode="average", reduction="sum", return_ranks=True
    )

    ground_truth_col = 0 if corruption_scheme == "h" else 2
    if filter_scores:
        extra_filter_triples = np.copy(triples["test"])
        extra_filter_triples[:, ground_truth_col] += 1
        triples_to_filter = (
            [extra_filter_triples]
            if extra_only
            else [triples["test"], extra_filter_triples]
        )
    else:
        triples_to_filter = None

    pipeline = AllScoresPipeline(
        test_bs,
        corruption_scheme,
        score_fn,
        evaluation,
        filter_triples=triples_to_filter,  # type: ignore
        return_scores=True,
        return_topk=True,
        k=10,
        window_size=1000,
        use_ipu_model=True,
    )
    out = pipeline()

    # Shuffle triples in same order of out["scores"]
    triple_reordered = torch.from_numpy(
        ds.triples["test"][partitioned_triple_set.triple_sort_idx[out["triple_idx"]]]
    )

    # All scores, computed on CPU
    if corruption_scheme == "t":
        cpu_scores = score_fn.score_tails(
            unsharded_entity_table[triple_reordered[:, 0]],
            triple_reordered[:, 1],
            unsharded_entity_table.unsqueeze(0),
        )
    else:
        cpu_scores = score_fn.score_heads(
            unsharded_entity_table.unsqueeze(0),
            triple_reordered[:, 1],
            unsharded_entity_table[triple_reordered[:, 2]],
        )

    pos_scores = score_fn.score_triple(
        unsharded_entity_table[triple_reordered[:, 0]],
        triple_reordered[:, 1],
        unsharded_entity_table[triple_reordered[:, 2]],
    ).flatten()
    # mask positive scores to compute metrics
    cpu_scores[
        torch.arange(cpu_scores.shape[0]), triple_reordered[:, ground_truth_col]
    ] = -torch.inf
    if filter_scores:
        filter_triples = torch.from_numpy(
            np.concatenate(triples_to_filter, axis=0)  # type: ignore
        )
        tr_filter = get_entity_filter(
            triple_reordered, filter_triples, corruption_scheme
        )
        cpu_scores[tr_filter[:, 0], tr_filter[:, 1]] = -torch.inf
        # check filters (for convenience, here h/t entities are non-repeating)
        if extra_only:
            assert torch.all(tr_filter[:, 0] == torch.arange(n_test_triple))
            assert torch.all(
                tr_filter[:, 1] == triple_reordered[:, ground_truth_col] + 1
            )
        else:
            assert torch.all(tr_filter[::2, 0] == tr_filter[1::2, 0])
            assert torch.all(tr_filter[::2, 0] == torch.arange(n_test_triple))
            assert torch.all(tr_filter[::2, 1] == triple_reordered[:, ground_truth_col])
            assert torch.all(
                tr_filter[1::2, 1] == triple_reordered[:, ground_truth_col] + 1
            )

    cpu_ranks = evaluation.ranks_from_scores(pos_scores, cpu_scores)
    # we allow for a off-by-one rank difference on at most 1% of triples,
    # due to rounding differences in CPU vs IPU score computations
    assert torch.all(torch.abs(cpu_ranks - out["ranks"]) <= 1)
    assert (cpu_ranks != out["ranks"]).sum() < n_test_triple / 100

    # restore positive scores
    cpu_scores[
        torch.arange(cpu_scores.shape[0]), triple_reordered[:, ground_truth_col]
    ] = pos_scores

    assert_close(cpu_scores, out["scores"], atol=1e-3, rtol=1e-4)
    assert torch.all(
        torch.topk(cpu_scores, k=pipeline.k, dim=-1).indices == out["topk_global_id"]
    )
