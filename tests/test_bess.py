import ctypes
from typing import Dict

import numpy as np
import poptorch
import pytest
import torch
from torch.testing import assert_close

from besskge.batch_sampler import RigidShardedBatchSampler
from besskge.bess import BessKGE, ScoreMovingBessKGE, EmbeddingMovingBessKGE
from besskge.dataset import KGDataset
from besskge.loss import LogSigmoidLoss
from besskge.negative_sampler import TripleBasedShardedNegativeSampler
from besskge.scoring import TransE
from besskge.sharding import PartitionedTripleSet, Sharding

seed = 1234
n_entity = 500
n_relation_type = 10
n_shard = 4
n_test_triple = 1000
batches_per_step = 3
shard_bs = 48
n_negative = 250
embedding_size = 128


@pytest.mark.parametrize("model", [ScoreMovingBessKGE, EmbeddingMovingBessKGE])
@pytest.mark.parametrize(
    "corruption_scheme, duplicate_batch", [("h", False), ("t", False), ("ht", True)]
)
@pytest.mark.parametrize("flat_negative_format", [True, False])
def test_inference(
    model: BessKGE,
    corruption_scheme: str,
    duplicate_batch: bool,
    flat_negative_format: bool,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    sharding = Sharding.create(n_entity, n_shard, seed=seed)

    entity_table = torch.randn(
        size=(sharding.n_shard, sharding.max_entity_per_shard, embedding_size)
    )
    relation_table = torch.randn(size=(n_relation_type, embedding_size))

    test_triples_h = np.random.randint(n_entity, size=(n_test_triple,))
    test_triples_t = np.random.randint(n_entity, size=(n_test_triple,))
    test_triples_r = np.random.randint(n_relation_type, size=(n_test_triple,))
    triples = {
        "test": np.stack([test_triples_h, test_triples_r, test_triples_t], axis=1)
    }

    neg_outer_shape = 1 if flat_negative_format else n_test_triple
    test_negative_heads = np.random.randint(
        n_entity, size=(neg_outer_shape, n_negative)
    )
    test_negative_tails = np.random.randint(
        n_entity, size=(neg_outer_shape, n_negative)
    )
    neg_heads = {"test": test_negative_heads}
    neg_tails = {"test": test_negative_tails}

    ds = KGDataset(
        n_entity=n_entity,
        n_relation_type=n_relation_type,
        entity_dict=None,
        relation_dict=None,
        type_offsets=None,
        triples=triples,
        neg_heads=neg_heads,
        neg_tails=neg_tails,
    )

    partitioned_triple_set = PartitionedTripleSet.create_from_dataset(
        ds, "test", sharding, partition_mode="ht_shardpair"
    )

    score_fn = TransE(negative_sample_sharing=flat_negative_format, scoring_norm=1)

    test_ns = TripleBasedShardedNegativeSampler(
        partitioned_triple_set.neg_heads,
        partitioned_triple_set.neg_tails,
        sharding,
        corruption_scheme=corruption_scheme,
        seed=seed,
        return_sort_idx=True,
    )

    test_bs = RigidShardedBatchSampler(
        partitioned_triple_set=partitioned_triple_set,
        negative_sampler=test_ns,
        shard_bs=shard_bs,
        batches_per_step=batches_per_step,
        seed=seed,
        hrt_freq_weighting=False,
        duplicate_batch=duplicate_batch,
        return_triple_idx=True,
    )

    test_dl = test_bs.get_dataloader(shuffle=False)

    ctypes.cdll.LoadLibrary("./custom_ops.so")
    options = poptorch.Options()
    options.replication_factor = sharding.n_shard
    options.deviceIterations(test_bs.batches_per_step)
    options.useIpuModel(True)
    options.outputMode(poptorch.OutputMode.All)

    # Define model with custom embedding tables
    inf_model = model(
        sharding=sharding,
        n_relation_type=ds.n_relation_type,
        embedding_size=embedding_size,
        negative_sampler=test_ns,
        entity_intializer=entity_table,
        relation_intializer=relation_table,
        score_fn=score_fn,
        return_scores=True,
    )
    # Load embedding tables
    inf_model.entity_embedding = torch.nn.Parameter(entity_table)
    inf_model.relation_embedding = torch.nn.Parameter(relation_table)

    ipu_inf_model = poptorch.inferenceModel(inf_model, options=options)

    ipu_inf_model.entity_embedding.replicaGrouping(
        poptorch.CommGroupType.NoGrouping,
        0,
        poptorch.VariableRetrievalMode.OnePerGroup,
    )

    # Compute true positive/negative scores
    test_h_embs = entity_table[
        sharding.entity_to_shard[test_triples_h], sharding.entity_to_idx[test_triples_h]
    ]
    test_rel_embs = relation_table[test_triples_r]
    test_t_embs = entity_table[
        sharding.entity_to_shard[test_triples_t], sharding.entity_to_idx[test_triples_t]
    ]
    neg_h_embs = entity_table[
        sharding.entity_to_shard[test_negative_heads],
        sharding.entity_to_idx[test_negative_heads],
    ]
    neg_t_embs = entity_table[
        sharding.entity_to_shard[test_negative_tails],
        sharding.entity_to_idx[test_negative_tails],
    ]

    true_positive_score = score_fn.score_triple(
        test_h_embs,
        test_rel_embs,
        test_t_embs,
    )
    true_neg_h_score = score_fn.score_heads(
        neg_h_embs,
        test_rel_embs,
        test_t_embs,
    )
    true_neg_t_score = score_fn.score_tails(
        test_h_embs,
        test_rel_embs,
        neg_t_embs,
    )

    # Check BESS inference results
    for batch in test_dl:
        triple_idx = batch.pop("triple_idx")
        triple_mask = batch.pop("triple_mask")
        negative_sort_idx = batch.pop("negative_sort_idx")
        res = ipu_inf_model(**{k: v.flatten(end_dim=1) for k, v in batch.items()})

        positive_score = res["positive_score"].reshape(
            batches_per_step, n_shard, n_shard, -1
        )

        negative_score = res["negative_score"].reshape(
            batches_per_step, n_shard, n_shard, -1, n_negative
        )

        negative_sort_idx = negative_sort_idx.reshape(
            batches_per_step, n_shard, n_shard, -1, n_negative
        ).type(torch.LongTensor)

        if test_bs.duplicate_batch:
            cutpoint = positive_score.shape[-1] // 2
            triple_idx = triple_idx[:, :, :, :cutpoint]
            positive_score = positive_score[:, :, :, :cutpoint]
            triple_mask = triple_mask[:, :, :, :cutpoint]
            negative_score_1, negative_score_2 = torch.split(
                negative_score, negative_score.shape[-2] // 2, dim=-2
            )
            negative_sort_idx_1, negative_sort_idx_2 = torch.split(
                negative_sort_idx, cutpoint, dim=-2
            )

        # Indices of triples in test_bs.triples accessed by the dataloader
        global_idx = triple_idx[triple_mask]
        # Discard padding triples
        positive_filtered = positive_score[triple_mask]
        # Use partitioned_triple_set.sort_idx to pass from indices in ds.triples to test_bs.triples
        assert_close(
            true_positive_score[partitioned_triple_set.triple_sort_idx][global_idx],
            positive_filtered,
        )

        if test_bs.duplicate_batch:
            # Discard padding triples
            neg_h_scores_filtered = negative_score_1[triple_mask]
            neg_t_scores_filtered = negative_score_2[triple_mask]
            negative_sort_idx_h = negative_sort_idx_1[triple_mask]
            negative_sort_idx_t = negative_sort_idx_2[triple_mask]
            # For each triple, use negative_sort_idx_h/t to pass from negative indices
            # in ds.neg_heads/tails to test_ns.padded_negatives
            assert_close(
                torch.take_along_dim(
                    true_neg_h_score[partitioned_triple_set.triple_sort_idx][
                        global_idx
                    ],
                    negative_sort_idx_h,
                    dim=-1,
                ),
                neg_h_scores_filtered,
            )
            assert_close(
                torch.take_along_dim(
                    true_neg_t_score[partitioned_triple_set.triple_sort_idx][
                        global_idx
                    ],
                    negative_sort_idx_t,
                    dim=-1,
                ),
                neg_t_scores_filtered,
            )
        else:
            negative_score_filtered = negative_score[triple_mask]
            negative_sort_idx = negative_sort_idx[triple_mask]
            true_neg_score = (
                true_neg_h_score
                if test_ns.corruption_scheme == "h"
                else true_neg_t_score
            )
            assert_close(
                torch.take_along_dim(
                    true_neg_score[partitioned_triple_set.triple_sort_idx][global_idx],
                    negative_sort_idx,
                    dim=-1,
                ),
                negative_score_filtered,
            )
