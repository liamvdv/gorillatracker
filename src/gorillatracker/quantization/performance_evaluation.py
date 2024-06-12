import os
from typing import Any, Dict, Union

import torch
import torch.nn as nn
from ai_edge_torch.model import TfLiteModel
from torch.fx import GraphModule

from gorillatracker.metrics import knn
from gorillatracker.model import BaseModule


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
    knn_number: int = 5,
) -> dict[str, Any]:

    model.eval()
    quantized_model = model.to(device)
    images = images.to(device)
    generated_image_embeddings = quantized_model(images)
    validation_labels = labels
    knn_results = knn(generated_image_embeddings, validation_labels, knn_number)
    return knn_results


@torch.no_grad()
def get_knn_accuracy_tflite(
    model: TfLiteModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    knn_number: int = 5,
) -> dict[str, Any]:

    generated_image_embeddings = model(images)
    generated_image_embeddings = torch.tensor(generated_image_embeddings)
    validation_labels = labels
    knn_results = knn(generated_image_embeddings, validation_labels, knn_number)
    return knn_results


def evaluate_model(
    model: Union[GraphModule, BaseModule, TfLiteModel],
    key: str,
    results: Dict[str, Any],
    validations_input_embeddings: torch.Tensor,
    validation_labels: torch.Tensor,
    model_path: str = "",
) -> None:
    if isinstance(model, TfLiteModel):
        results[key] = dict()
        results[key]["size_of_model_in_mb"] = os.path.getsize(model_path) / 1e6
        results[key]["knn1"] = get_knn_accuracy_tflite(
            model=model,
            images=validations_input_embeddings,
            labels=validation_labels,
            knn_number=1,
        )

        results[key]["knn5"] = get_knn_accuracy_tflite(
            model=model,
            images=validations_input_embeddings,
            labels=validation_labels,
            knn_number=5,
        )
        return

    results[key]["size_of_model_in_mb"] = size_of_model_in_mb(model)
    results[key]["knn1"] = get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        device=torch.device("cpu"),
        knn_number=1,
    )

    results[key]["knn5"] = get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        device=torch.device("cpu"),
        knn_number=5,
    )
