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

test_triples_h = np.random.randint(n_entity, size=(n_test_triple,))
test_triples_t = np.random.randint(n_entity, size=(n_test_triple,))
test_triples_r = np.random.randint(n_relation_type, size=(n_test_triple,))
triples = {"test": np.stack([test_triples_h, test_triples_r, test_triples_t], axis=1)}


@pytest.mark.parametrize("corruption_scheme", ["h", "t"])
@pytest.mark.parametrize("compute_metrics", [True, False])
def test_all_scores_pipeline(corruption_scheme: str, compute_metrics: bool) -> None:
    ds = KGDataset(
        n_entity=n_entity,
        n_relation_type=n_relation_type,
        entity_dict=None,
        relation_dict=None,
        type_offsets=None,
        triples=triples,
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

    if compute_metrics:
        evaluation = Evaluation(
            ["mrr", "hits@10"], mode="average", reduction="sum", return_ranks=True
        )
    else:
        evaluation = None

    pipeline = AllScoresPipeline(
        test_bs,
        corruption_scheme,
        score_fn,
        evaluation,
        return_scores=True,
        windows_size=1000,
        use_ipu_model=True,
    )
    out = pipeline()

    # Shuffle triples in same order of out["scores"]
    triple_reordered = torch.from_numpy(
        ds.triples["test"][partitioned_triple_set.triple_sort_idx[out["triple_idx"]]]
    )

    # Real scores
    if corruption_scheme == "t":
        real_scores = score_fn.score_tails(
            unsharded_entity_table[triple_reordered[:, 0]],
            triple_reordered[:, 1],
            unsharded_entity_table.unsqueeze(0),
        )
    else:
        real_scores = score_fn.score_heads(
            unsharded_entity_table.unsqueeze(0),
            triple_reordered[:, 1],
            unsharded_entity_table[triple_reordered[:, 2]],
        )

    assert_close(real_scores, out["scores"], atol=1e-3, rtol=1e-4)

    if evaluation:
        ground_truth_col = 0 if corruption_scheme == "h" else 2
        real_scores_masked = torch.clone(real_scores)
        pos_scores = real_scores_masked[
            torch.arange(real_scores.shape[0]), triple_reordered[:, ground_truth_col]
        ]
        real_scores_masked[
            torch.arange(real_scores.shape[0]), triple_reordered[:, ground_truth_col]
        ] = -torch.inf

        real_ranks = evaluation.ranks_from_scores(pos_scores, real_scores_masked)

        # we allow for a off-by-one rank difference on at most 1% of triples,
        # due to rounding differences in CPU vs IPU score computations
        assert torch.all(torch.abs(real_ranks - out["ranks"]) <= 1)
        assert (real_ranks != out["ranks"]).sum() < n_test_triple / 100
