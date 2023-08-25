# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Functions for computing the batch loss based on the scores of positive
and negative samples.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseLossFunction(torch.nn.Module, ABC):
    """
    Base class for a loss function.

    Losses are always computed in FP32.
    """

    #: Use self-adversarial weighting of negative samples.
    negative_adversarial_sampling: bool
    #: Reciprocal temperature of self-adversarial weighting
    negative_adversarial_scale: torch.Tensor
    #: Loss scaling factor, might be needed when using FP16 weights
    loss_scale: torch.Tensor

    def get_negative_weights(self, negative_score: torch.Tensor) -> torch.Tensor:
        """
        Construct weights of negative samples, based on their score.

        :param negative_score: : (batch_size, n_negative)
            Scores of negative samples.

        :return: shape: (batch_size, n_negative)
            if :attr:`BaseLossFunction.negative_adversarial_sampling` else ()
            Weights of negative samples.
        """
        if self.negative_adversarial_sampling:
            negative_weights = torch.nn.functional.softmax(
                self.negative_adversarial_scale.to(negative_score.device)
                * negative_score,
                dim=-1,
            ).detach()
        else:
            negative_weights = torch.tensor(
                1.0 / negative_score.shape[-1],
                requires_grad=False,
                device=negative_score.device,
            )
        return negative_weights

    @abstractmethod
    def forward(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        triple_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute batch loss.

        :param positive_score: shape: (batch_size,)
            Scores of positive triples.
        :param negative_score: shape: (batch_size, n_negative)
            Scores of negative triples.
        :param triple_weight: shape: (batch_size,) or ()
            Weights of positive triples.

        :return:
            The batch loss.
        """
        raise NotImplementedError


class MarginBasedLossFunction(BaseLossFunction, ABC):
    """
    Base class for margin-based loss functions.
    """

    def __init__(
        self,
        margin: float,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        loss_scale: float = 1.0,
    ) -> None:
        """
        Initialize margin-based loss function.

        :param margin:
            The margin to be used in the loss computation.
        :param negative_adversarial_sampling:
            see :class:`BaseLossFunction`
        :param negative_adversarial_scale:
            see :class:`BaseLossFunction`
        :param loss_scale:
            see :class:`BaseLossFunction`
        """
        super(MarginBasedLossFunction, self).__init__()
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.negative_adversarial_scale = torch.tensor(
            negative_adversarial_scale, dtype=torch.float32
        )
        self.loss_scale = torch.tensor(loss_scale, dtype=torch.float32)
        self.margin: torch.Tensor = torch.tensor(margin, dtype=torch.float32)


class LogSigmoidLoss(MarginBasedLossFunction):
    """
    The log-sigmoid loss function (see :cite:p:`RotatE`).
    """

    # docstr-coverage: inherited
    def forward(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        triple_weight: torch.Tensor,
    ) -> torch.Tensor:
        negative_score_weights = self.get_negative_weights(negative_score)
        positive_score_logs = torch.nn.functional.logsigmoid(
            positive_score + self.margin.to(positive_score.device)
        )
        negative_score_logs = torch.nn.functional.logsigmoid(
            -negative_score - self.margin.to(negative_score.device)
        )
        negative_score_reduced = torch.sum(
            negative_score_weights * negative_score_logs, dim=-1
        )
        loss = torch.tensor(
            -0.5, dtype=torch.float32, device=triple_weight.device
        ) * torch.sum(triple_weight * (positive_score_logs + negative_score_reduced))
        return self.loss_scale.to(device=loss.device) * loss


class MarginRankingLoss(MarginBasedLossFunction):
    """
    The margin ranking (or pairwise hinge) loss function.
    """

    def __init__(
        self,
        margin: float,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        loss_scale: float = 1.0,
        activation_function: str = "relu",
    ) -> None:
        """
        Initialize margin ranking loss function.

        :param margin:
            see :meth:`MarginBasedLossFunction.__init__`
        :param negative_adversarial_sampling:
            see :class:`BaseLossFunction`
        :param negative_adversarial_scale:
            see :class:`BaseLossFunction`
        :param loss_scale:
            see :class:`BaseLossFunction`
        :param activation_function:
            The activation function in loss computation. Default: "relu".
        """
        super(MarginRankingLoss, self).__init__(
            margin,
            negative_adversarial_sampling,
            negative_adversarial_scale,
            loss_scale,
        )
        if activation_function == "relu":
            self.activation = torch.nn.functional.relu
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported"
                " for MarginRankingLoss"
            )

    # docstr-coverage: inherited
    def forward(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        triple_weight: torch.Tensor,
    ) -> torch.Tensor:
        negative_score_weights = self.get_negative_weights(negative_score)
        combined_score = self.activation(
            negative_score
            - positive_score.unsqueeze(1)
            + self.margin.to(positive_score.device)
        )
        combined_score_reduced = torch.sum(
            negative_score_weights * combined_score, dim=-1
        )
        loss = torch.sum(triple_weight * combined_score_reduced)
        return self.loss_scale.to(device=loss.device) * loss


class SampledSoftmaxCrossEntropyLoss(BaseLossFunction):
    """
    The sampled softmax cross-entropy loss (see :cite:p:`large_vocabulary` and
    :cite:p:`BESS`).
    """

    def __init__(
        self,
        n_entity: int,
        loss_scale: float = 1.0,
    ) -> None:
        """
        Initialize the sampled softmax cross-entropy loss.

        :param n_entity:
            The total number of entities in the knowledge graph.
        :param loss_scale:
            see :class:`BaseLossFunction`
        """
        super(SampledSoftmaxCrossEntropyLoss, self).__init__()
        self.negative_adversarial_sampling = False
        self.negative_adversarial_scale = torch.tensor(0.0, dtype=torch.float32)
        self.loss_scale = torch.tensor(loss_scale, dtype=torch.float32)
        self.n_entity = n_entity

    # docstr-coverage: inherited
    def forward(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        triple_weight: torch.Tensor,
    ) -> torch.Tensor:
        # Before computing the cross entropy, scores are adjusted by
        # `log(1 / E(count(candidate == class)))`, which is
        # constant over all negative classes and zero for the target class.
        negative_score += torch.tensor(
            np.log(self.n_entity - 1) - np.log(negative_score.shape[1]),
            device=negative_score.device,
            dtype=negative_score.dtype,
        )

        cross_entropy_score = torch.nn.functional.cross_entropy(
            input=torch.concat([positive_score.unsqueeze(1), negative_score], dim=-1),
            target=torch.full_like(
                positive_score,
                fill_value=0,
                dtype=torch.int32,
                device=positive_score.device,
            ),
            reduction="none",
        )
        loss = torch.sum(triple_weight * cross_entropy_score)

        return self.loss_scale.to(device=loss.device) * loss
