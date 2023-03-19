# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from abc import ABC, abstractmethod

import torch


class BaseLossFunction(ABC):
    """
    Base class for a loss function.
    """

    def __init__(
        self,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize loss function.

        :param negative_adversarial_sampling:
            Use self-adversarial weighting of negative samples.
        :param negative_adversarial_scale:
            Reciprocal temperature of self-adversarial weighting, defaults to 1.0.
        """
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.negative_adversarial_scale = negative_adversarial_scale

    def get_negative_weights(
        self, negative_score: torch.FloatTensor
    ) -> torch.FloatTensor:
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
    def compute_loss(
        self,
        positive_score: torch.FloatTensor,
        negative_score: torch.FloatTensor,
        triple_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
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
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize margin-based loss function.

        :param margin:
            Margin to be used in the loss computation.
        :param negative_adversarial_sampling:
            see :meth:`BaseLossFunction.__init__`
        :param negative_adversarial_scale:
            see :meth:`BaseLossFunction.__init__`
        """
        super(MarginBasedLossFunction, self).__init__(
            negative_adversarial_sampling, negative_adversarial_scale, *args, **kwargs
        )
        self.margin = margin


class LogSigmoidLoss(MarginBasedLossFunction):
    """
    The log-sigmoid loss function (see [...]).
    """

    # docstr-coverage: inherited
    def compute_loss(
        self,
        positive_score: torch.FloatTensor,
        negative_score: torch.FloatTensor,
        triple_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
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
        *args,
        **kwargs,
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
            *args,
            **kwargs,
        )
        if activation_function == "relu":
            self.activation = torch.nn.functional.relu
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported for MarginRankingLoss"
            )

    # docstr-coverage: inherited
    def compute_loss(
        self,
        positive_score: torch.FloatTensor,
        negative_score: torch.FloatTensor,
        triple_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        negative_score_weights = self.get_negative_weights(negative_score)
        combined_score = self.activation(
            negative_score - positive_score.unsqueeze(1) + self.margin
        )
        combined_score_reduced = torch.sum(
            negative_score_weights * combined_score, dim=-1
        )
        return torch.sum(triple_weight * combined_score_reduced)
