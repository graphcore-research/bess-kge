# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
High-level APIs for training/inference with BESS.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import poptorch
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from besskge.batch_sampler import ShardedBatchSampler
from besskge.bess import AllScoresBESS
from besskge.metric import Evaluation
from besskge.negative_sampler import PlaceholderNegativeSampler
from besskge.scoring import BaseScoreFunction
from besskge.utils import get_entity_filter


class AllScoresPipeline(torch.nn.Module):
    """
    Pipeline to compute scores of (h, r, ?) / (?, r, t) queries against all entities
    in the KG, and related prediction metrics.
    It supports filtering out the scores of specific completions that appear in a given
    set of triples.

    To be used in combination with a batch sampler based on a
    "h_shard"/"t_shard"-partitioned triple set.
    """

    def __init__(
        self,
        batch_sampler: ShardedBatchSampler,
        corruption_scheme: str,
        score_fn: BaseScoreFunction,
        evaluation: Optional[Evaluation] = None,
        filter_triples: Optional[List[Union[torch.Tensor, NDArray[np.int32]]]] = None,
        return_scores: bool = False,
        return_topk: bool = False,
        k: int = 10,
        window_size: int = 1000,
        use_ipu_model: bool = False,
    ) -> None:
        """
        Initialize pipeline.

        :param batch_sampler:
            Batch sampler, based on a
            "h_shard"/"t_shard"-partitioned triple set.
        :param corruption_scheme:
            Set to "t" to score (h, r, ?) completions, or to
            "h" to score (?, r, t) completions.
        :param score_fn:
            The trained scoring function.
        :param evaluation:
            Evaluation module, for computing metrics.
            Default: None.
        :param filter_triples:
            The set of all triples whose scores need to be filtered.
            The triples passed here must have GLOBAL IDs for head/tail
            entities. Default: None.
        :param return_scores:
            If True, store and return scores of all queries' completions
            (with filters applied, if specified).
            For large number of queries/entities, this can cause the host
            to go OOM.
            Default: False.
        :param return_topk:
            If True, return for each query the global IDs of the most likely
            completions, after filtering out the scores of `filter_triples`.
            Default: False.
        :param k:
            If `return_topk` is set to True, for each query return the
            top-k most likely predictions (after filtering). Default: 10.
        :param window_size:
            Size of the sliding window, namely the number of negative entities
            scored against each query at each step on IPU and returned to host.
            Should be decreased with large batch sizes, to avoid an OOM error.
            Default: 1000.
        :param use_ipu_model:
            Run pipeline on IPU Model instead of actual hardware. Default: False.
        """
        super().__init__()
        self.batch_sampler = batch_sampler
        if not (evaluation or return_scores):
            raise ValueError(
                "Nothing to return. Provide `evaluation` or set `return_scores=True`"
            )
        if corruption_scheme not in ["h", "t"]:
            raise ValueError("corruption_scheme needs to be either 'h' or 't'")
        if (
            corruption_scheme == "h"
            and self.batch_sampler.triple_partition_mode != "t_shard"
        ):
            raise ValueError(
                "Corruption scheme 'h' requires 't-shard'-partitioned triples"
            )
        elif (
            corruption_scheme == "t"
            and self.batch_sampler.triple_partition_mode != "h_shard"
        ):
            raise ValueError(
                "Corruption scheme 't' requires 'h-shard'-partitioned triples"
            )
        self.candidate_sampler = PlaceholderNegativeSampler(
            corruption_scheme=corruption_scheme
        )
        self.score_fn = score_fn
        self.evaluation = evaluation
        self.return_scores = return_scores
        self.return_topk = return_topk
        self.k = k
        self.window_size = window_size
        self.corruption_scheme = corruption_scheme
        self.bess_module = AllScoresBESS(
            self.candidate_sampler, self.score_fn, self.window_size
        )

        inf_options = poptorch.Options()
        inf_options.replication_factor = self.bess_module.sharding.n_shard
        inf_options.deviceIterations(self.batch_sampler.batches_per_step)
        inf_options.outputMode(poptorch.OutputMode.All)
        if use_ipu_model:
            inf_options.useIpuModel(True)
        self.dl = self.batch_sampler.get_dataloader(options=inf_options, shuffle=False)

        self.poptorch_module = poptorch.inferenceModel(
            self.bess_module, options=inf_options
        )
        self.poptorch_module.entity_embedding.replicaGrouping(
            poptorch.CommGroupType.NoGrouping,
            0,
            poptorch.VariableRetrievalMode.OnePerGroup,
        )
        self.filter_triples: Optional[torch.Tensor] = None
        if filter_triples:
            # Reconstruct global IDs for all entities in triples
            local_id_col = (
                0 if self.batch_sampler.triple_partition_mode == "h_shard" else 2
            )
            triple_shard_offset = np.concatenate(
                [np.array([0]), np.cumsum(batch_sampler.triple_counts)]
            )
            global_id_triples = []
            for i in range(len(triple_shard_offset) - 1):
                shard_triples = np.copy(
                    batch_sampler.triples[
                        triple_shard_offset[i] : triple_shard_offset[i + 1]
                    ]
                )
                shard_triples[
                    :, local_id_col
                ] = self.bess_module.sharding.shard_and_idx_to_entity[i][
                    shard_triples[:, local_id_col]
                ]
                global_id_triples.append(shard_triples)
            self.triples = torch.from_numpy(np.concatenate(global_id_triples, axis=0))
            self.filter_triples = torch.concat(
                [
                    tr if isinstance(tr, torch.Tensor) else torch.from_numpy(tr)
                    for tr in filter_triples
                ],
                dim=0,
            )

    def forward(self) -> Dict[str, Any]:
        """
        Compute scores of all completions and (possibly) metrics.

        :return:
            Scores, metrics, and (if provided in batch sampler) IDs
            of inference triples (wrt partitioned_triple_set.triples)
            to order results.
        """
        scores = []
        ids = []
        metrics = []
        ranks = []
        topk_ids = []
        n_triple = 0
        for batch in tqdm(iter(self.dl)):
            triple_mask = batch.pop("triple_mask")
            if (
                self.candidate_sampler.corruption_scheme == "h"
                and "head" in batch.keys()
            ):
                ground_truth = batch.pop("head")
            elif (
                self.candidate_sampler.corruption_scheme == "t"
                and "tail" in batch.keys()
            ):
                ground_truth = batch.pop("tail")
            if self.batch_sampler.return_triple_idx:
                triple_id = batch.pop("triple_idx")
                ids.append(triple_id[triple_mask])
            n_triple += triple_mask.sum()

            batch_res = []
            batch_idx = []
            for i in range(self.bess_module.n_step):
                step = (
                    torch.tensor([i], dtype=torch.int32)
                    .broadcast_to(
                        (
                            self.bess_module.sharding.n_shard
                            * self.batch_sampler.batches_per_step,
                            1,
                        )
                    )
                    .contiguous()
                )
                ent_slice = torch.minimum(
                    i * self.bess_module.window_size
                    + torch.arange(self.bess_module.window_size),
                    torch.tensor(self.bess_module.sharding.max_entity_per_shard - 1),
                )
                # Global indices of entities scored in the step
                batch_idx.append(
                    self.bess_module.sharding.shard_and_idx_to_entity[
                        :, ent_slice
                    ].flatten()
                )
                inp = {k: v.flatten(end_dim=1) for k, v in batch.items()}
                inp.update(dict(step=step))
                batch_res.append(self.poptorch_module(**inp))
            batch_scores = torch.concat(batch_res, dim=-1)
            # Filter out padding scores
            batch_scores_filt = batch_scores[triple_mask.flatten()][
                :, np.unique(np.concatenate(batch_idx), return_index=True)[1]
            ][:, : self.bess_module.sharding.n_entity]
            if ground_truth is not None:
                # Scores of positive triples
                true_scores = batch_scores_filt[
                    torch.arange(batch_scores_filt.shape[0]),
                    ground_truth[triple_mask],
                ]
            if self.filter_triples is not None:
                # Filter for triples in batch
                batch_filter = get_entity_filter(
                    self.triples[triple_id[triple_mask]],
                    self.filter_triples,
                    filter_mode=self.corruption_scheme,
                )
                batch_scores_filt[batch_filter[:, 0], batch_filter[:, 1]] = -torch.inf

            if self.evaluation:
                assert (
                    ground_truth is not None
                ), "Evaluation requires providing ground truth entities"
                # If not already masked, mask scores of true triples
                # to compute metrics
                batch_scores_filt[
                    torch.arange(batch_scores_filt.shape[0]),
                    ground_truth[triple_mask],
                ] = -torch.inf
                batch_ranks = self.evaluation.ranks_from_scores(
                    true_scores, batch_scores_filt
                )
                metrics.append(self.evaluation.dict_metrics_from_ranks(batch_ranks))
                if self.evaluation.return_ranks:
                    ranks.append(batch_ranks)
            if ground_truth is not None:
                # Restore positive scores in the returned scores
                batch_scores_filt[
                    torch.arange(batch_scores_filt.shape[0]),
                    ground_truth[triple_mask],
                ] = true_scores
            if self.return_scores:
                scores.append(batch_scores_filt)
            if self.return_topk:
                topk_ids.append(torch.topk(batch_scores_filt.to(torch.float32), k=self.k, dim=-1).indices)

        out = dict()
        if scores:
            out["scores"] = torch.concat(scores, dim=0)
        if topk_ids:
            out["topk_global_id"] = torch.concat(topk_ids, dim=0)
        if ids:
            out["triple_idx"] = torch.concat(ids, dim=0)
        if self.evaluation:
            final_metrics = dict()
            for m in metrics[0].keys():
                # Reduce metrics over all batches
                final_metrics[m] = self.evaluation.reduction(
                    torch.concat([met[m].reshape(-1) for met in metrics])
                )
            out["metrics"] = final_metrics  # type: ignore
            # Average metrics over all triples
            out["metrics_avg"] = {
                m: v.sum() / n_triple for m, v in final_metrics.items()
            }  # type: ignore
            if ranks:
                out["ranks"] = torch.concat(ranks, dim=0)

        return out
