from typing import Any

import torch
import torch.nn as nn

import gorillatracker.type_helper as gtypes


class OfflineResponseBasedLoss(nn.Module):
    def __init__(self, teacher_model_wandb_link: str):
        from gorillatracker.utils.embedding_generator import get_model_for_run_url

        assert teacher_model_wandb_link != "", "Teacher model link is not provided"
        self.teacher_model = get_model_for_run_url(teacher_model_wandb_link)
        self.teacher_model.eval()
        self.loss = nn.MSELoss()

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, images: torch.Tensor, **kwargs: Any
    ) -> gtypes.LossPosNegDist:
        teacher_embeddings = self.teacher_model(images)
        return (
            self.loss(embeddings, teacher_embeddings),
            torch.Tensor([-1.0]),
            torch.Tensor([-1.0]),
        )  # dummy values for pos/neg distances
