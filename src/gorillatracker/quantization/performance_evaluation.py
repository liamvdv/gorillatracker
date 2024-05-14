import os

import torch
import torch.nn as nn

from gorillatracker.metrics import knn_naive


def size_of_model_in_mb(model: nn.Module) -> float:
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return model_size


@torch.no_grad()
def get_knn_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    knn: int = 5,
) -> dict:

    model.eval()
    quantized_model = model.to(device)
    images = images.to(device)
    generated_image_embeddings = quantized_model(images)
    validation_labels = labels
    knn_results = knn_naive(generated_image_embeddings, validation_labels, knn)
    return knn_results
