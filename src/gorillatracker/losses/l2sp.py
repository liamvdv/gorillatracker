from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

import gorillatracker.type_helper as gtypes
from gorillatracker.utils.l2sp_regularisation import L2_SP


class L2SPRegularization_Wrapper(nn.Module):
    """Wrapper that adds L2SP regularization to any loss"""

    def __init__(
        self,
        loss: nn.Module,
        model: nn.Module,
        path_to_pretrained_weights: str,
        alpha: float,
        beta: float,
        log_func: Callable[[str, float], None] = lambda x, y: None,
    ):
        super().__init__()
        assert path_to_pretrained_weights is not None, "Path to pretrained weights must be provided"
        self.loss = loss
        self.model = model
        self.l2sp_loss = L2_SP(model, path_to_pretrained_weights, alpha, beta)
        self.log = log_func

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> gtypes.LossPosNegDist:
        standard_loss, anchor_positive_dist_mean, anchor_negative_dist_mean = self.loss(*args, **kwargs)
        l2sp_loss = self.l2sp_loss(self.model)
        if type(l2sp_loss) == torch.Tensor:
            self.log("l2_sp", l2sp_loss.item())
        else:
            self.log("l2_sp", l2sp_loss)
        return standard_loss + l2sp_loss, anchor_positive_dist_mean, anchor_negative_dist_mean
