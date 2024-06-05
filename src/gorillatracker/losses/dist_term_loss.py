from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

from gorillatracker import type_helper as gtypes

from .arcface_loss import ArcFaceLoss
from .triplet_loss import TripletLossOnline


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    # formula: 1 - cosine similarity = 1 - (A.B / |A||B|) (range: [0, 2]) -> divide by 2 to get range [0, 1]

    dot_product = torch.mm(embeddings, embeddings.t())  # shape: (batch_size, batch_size)
    norm = torch.sqrt(torch.diag(dot_product))  # shape: (batch_size,)
    norm = norm.unsqueeze(0) * norm.unsqueeze(1)  # shape: (batch_size, batch_size)

    return (1 - dot_product / norm) / 2


class CombinedLoss(nn.Module):
    def __init__(
        self,
        arcface_loss: ArcFaceLoss,
        triplet_loss: TripletLossOnline,
        lambda_: float = 0.5,
        log_func: Callable[[str, float], None] = lambda x, y: None,
    ):
        super().__init__()
        self.arcface = arcface_loss
        self.triplet = triplet_loss
        self.lambda_ = lambda_
        self.log_func = log_func

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> gtypes.LossPosNegDist:
        loss_arcface, _, _ = self.arcface(*args, **kwargs)
        loss_triplet, pos_dist, neg_dist = self.triplet(*args, **kwargs, dist_calc=cosine_distance_matrix)
        loss_triplet = torch.clamp(loss_triplet, min=0.0, max=1.0).to(loss_arcface.device)

        self.log_func("arcface", loss_arcface.item())
        self.log_func("triplet", loss_triplet.item())
        return loss_arcface + self.lambda_ * loss_triplet, pos_dist, neg_dist
