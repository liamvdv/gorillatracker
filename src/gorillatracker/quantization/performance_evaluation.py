import os
from typing import Any, Dict, Union

import pandas as pd
import torch
import torch.nn as nn
from ai_edge_torch.model import TfLiteModel
from torch.fx import GraphModule
import uuid

from gorillatracker.metrics import knn
from gorillatracker.model.base_module import BaseModule
from gorillatracker.utils.labelencoder import LinearSequenceEncoder


def size_of_model_in_mb(model: nn.Module) -> float:
    name = str(uuid.uuid4()) + ".p"
    torch.save(model.state_dict(), name)
    model_size = os.path.getsize(name) / 1e6
    os.remove(name)
    return model_size


def process_images_in_batches(quantized_model, images, batch_size=100):
    generated_image_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]

        # If the batch is smaller than batch_size, pad it with dummy images
        if len(batch) < batch_size:
            dummy_images = torch.zeros_like(batch[0]).unsqueeze(0).repeat(batch_size - len(batch), 1, 1, 1)
            batch = torch.cat([batch, dummy_images], dim=0)

        # Process the batch
        batch_embeddings = quantized_model(batch)

        # If we added dummy images, remove their embeddings
        if i + batch_size > len(images):
            batch_embeddings = batch_embeddings[: len(images) - i]

        generated_image_embeddings.append(batch_embeddings)

    # Concatenate all embeddings
    return torch.cat(generated_image_embeddings)


@torch.no_grad()
def get_knn_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    ids: list[str],
    device: torch.device = torch.device("cpu"),
    knn_number: int = 5,
    cross_video: bool = False,
) -> dict[str, Any]:

    quantized_model = model.to(device)
    images = images.to(device)

    generated_image_embeddings = process_images_in_batches(quantized_model, images).to("cpu")

    validation_labels = labels

    le = LinearSequenceEncoder()
    # convert the labels tensor to a list of ints
    encoded_labels = le.encode_list(validation_labels.tolist())

    df_data = []
    for embedding, label, id, encoded_label in zip(generated_image_embeddings, validation_labels, ids, encoded_labels):
        df_data.append(
            {"embedding": embedding, "label": label, "id": id, "partition": "val", "encoded_label": encoded_label}
        )

    data = pd.DataFrame(df_data)

    knn_results = knn(data, k=knn_number, use_crossvideo_positives=cross_video)
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
    data = pd.DataFrame(
        {  # TODO(rob2u): IDK if this works ask kajo to test and fix together
            "embeddings": generated_image_embeddings,
            "labels": validation_labels,
            "id": range(len(validation_labels)),  # HACK
            "partition": "val",
            "dataset": "CXL",
        }
    )

    knn_results = knn(data, k=knn_number)
    return knn_results


def evaluate_model(
    model: Union[GraphModule, BaseModule, TfLiteModel],
    key: str,
    results: Dict[str, Any],
    validations_input_embeddings: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_ids: list[str],
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
    results[key] = dict()
    results[key]["size_of_model_in_mb"] = size_of_model_in_mb(model)
    results[key]["knn1"] = get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        ids=validation_ids,
        device=torch.device("cuda"),
        knn_number=1,
    )

    results[key]["knn1_crossvideo"] = get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        ids=validation_ids,
        device=torch.device("cuda"),
        knn_number=1,
        cross_video=True,
    )

    results[key]["knn5"] = get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        ids=validation_ids,
        device=torch.device("cuda"),
        knn_number=5,
    )

    results[key]["knn5_crossvideo"] = get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        device=torch.device("cuda"),
        ids=validation_ids,
        knn_number=5,
        cross_video=True,
    )
