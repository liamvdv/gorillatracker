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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, images: torch.Tensor) -> gtypes.LossPosNegDist:
        teacher_embeddings = self.teacher_model(images)
        return (
            self.loss(embeddings, teacher_embeddings),
            torch.Tensor([-1.0]),
            torch.Tensor([-1.0]),
        )  # dummy values for pos/neg distances
