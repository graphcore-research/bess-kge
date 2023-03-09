from abc import ABC, abstractmethod
import torch


class BaseLossFunction(ABC):
    def __init__(
        self,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.negative_adversarial_scale = negative_adversarial_scale

    def get_negative_weights(
        self, negative_score: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.negative_adversarial_sampling:
            negative_weights = torch.nn.functional.softmax(
                self.negative_adversarial_scale * negative_score, dim=-1
            ).detach()
        else:
            negative_weights = torch.tensor(1.0 / negative_score.shape[-1])
        return negative_weights

    @abstractmethod
    def compute_loss(
        self,
        positive_score: torch.FloatTensor,
        negative_score: torch.FloatTensor,
        triple_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class MarginBasedLossFunction(BaseLossFunction, ABC):
    def __init__(
        self,
        margin: float,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        *args,
        **kwargs,
    ):
        super(MarginBasedLossFunction, self).__init__(
            negative_adversarial_sampling, negative_adversarial_scale, *args, **kwargs
        )
        self.margin = margin


class LogSigmoidLoss(MarginBasedLossFunction):
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
    def __init__(
        self,
        margin: float,
        negative_adversarial_sampling: bool,
        negative_adversarial_scale: float = 1.0,
        activation_function: str = "relu",
        *args,
        **kwargs,
    ):
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
            raise NotImplementedError(
                f"Activation function {activation_function} not supported for MarginRankingLoss"
            )

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
