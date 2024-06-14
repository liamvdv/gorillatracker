from typing import Any

import torch
import torch.nn as nn

import gorillatracker.type_helper as gtypes


class OfflineResponseBasedLoss(nn.Module):
    def __init__(self, teacher_model_wandb_link: str):
        from gorillatracker.utils.embedding_generator import get_model_for_run_url

        super().__init__()
        assert teacher_model_wandb_link != "", "Teacher model link is not provided"
        self.teacher_model = get_model_for_run_url(teacher_model_wandb_link)
        self.teacher_model.eval()

        # Set requires_grad to False for all parameters of the teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

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


if __name__ == "__main__":
    # Test the OfflineResponseBasedLoss
    teacher_model_wandb_link = "https://wandb.ai/gorillas/Embedding-SwinV2-SSL-Face/runs/mqhtj5r5"
    offline_response_based_loss = OfflineResponseBasedLoss(teacher_model_wandb_link)

    embeddings = torch.randn(10, 256)
    labels = torch.randint(0, 10, (10,))
    images = torch.randn(10, 3, 192, 192)

    loss, pos_dist, neg_dist = offline_response_based_loss(embeddings, labels, images)
    print(loss, pos_dist, neg_dist)
