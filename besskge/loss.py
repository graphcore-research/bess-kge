# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod

import torch


class BaseLossFunction(ABC):
    """
    Base class for a loss function.
    """

    # Use self-adversarial weighting of negative samples.
    negative_adversarial_sampling: bool
    # Reciprocal temperature of self-adversarial weighting
    negative_adversarial_scale: float

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
                self.negative_adversarial_scale * negative_score, dim=-1
            ).detach()
        else:
            negative_weights = torch.tensor(
                1.0 / negative_score.shape[-1], requires_grad=False
            )
        return negative_weights

    @abstractmethod
    def __call__(
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


class MarginBasedLossFunction(BaseLossFunction):
    """
    Base class for margin-based loss functions.
    """

    def __init__(
        self,
        margin: float,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
    ) -> None:
        """
        Initialize margin-based loss function.

        :param margin:
            Margin to be used in the loss computation.
        :param negative_adversarial_sampling:
            see :class:`BaseLossFunction`
        :param negative_adversarial_scale:
            see :class:`BaseLossFunction`
        """
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.negative_adversarial_scale = negative_adversarial_scale
        self.margin = margin


class LogSigmoidLoss(MarginBasedLossFunction):
    """
    The log-sigmoid loss function (see [...]).
    """

    # docstr-coverage: inherited
    def __call__(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        triple_weight: torch.Tensor,
    ) -> torch.Tensor:
        negative_score_weights = self.get_negative_weights(negative_score)
        positive_score_logs = torch.nn.functional.logsigmoid(
            positive_score + self.margin
        )
        negative_score_logs = torch.nn.functional.logsigmoid(
            -negative_score - self.margin
        )
        negative_score_reduced = torch.sum(
            negative_score_weights * negative_score_logs, dim=-1
        )
        return -0.5 * torch.sum(
            triple_weight * (positive_score_logs + negative_score_reduced)
        )


class MarginRankingLoss(MarginBasedLossFunction):
    """
    The margin ranking (or pairwise hinge) loss function (see [...]).
    """

    def __init__(
        self,
        margin: float,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        activation_function: str = "relu",
    ) -> None:
        """
        Initialize margin ranking loss function.

        :param margin:
            see :meth:`MarginBasedLossFunction.__init__`
        :param negative_adversarial_sampling:
            see :meth:`BaseLossFunction.__init__`
        :param negative_adversarial_scale:
            see :meth:`BaseLossFunction.__init__`
        :param activation_function:
            Activation function in loss computation, defaults to "relu".
        """
        super(MarginRankingLoss, self).__init__(
            margin,
            negative_adversarial_sampling,
            negative_adversarial_scale,
        )
        if activation_function == "relu":
            self.activation = torch.nn.functional.relu
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported"
                " for MarginRankingLoss"
            )

    # docstr-coverage: inherited
    def __call__(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        triple_weight: torch.Tensor,
    ) -> torch.Tensor:
        negative_score_weights = self.get_negative_weights(negative_score)
        combined_score = self.activation(
            negative_score - positive_score.unsqueeze(1) + self.margin
        )
        combined_score_reduced = torch.sum(
            negative_score_weights * combined_score, dim=-1
        )
        return torch.sum(triple_weight * combined_score_reduced)
