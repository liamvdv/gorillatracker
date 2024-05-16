from typing import Union

import torch
from torch.fx import GraphModule

import gorillatracker.quantization.performance_evaluation as performance_evaluation
import gorillatracker.quantization.quantization_functions as quantization_functions
from gorillatracker.datasets.cxl import CXLDataset
from gorillatracker.model import BaseModule
from gorillatracker.quantization.utils import get_model_input, log_model_to_file
from gorillatracker.utils.embedding_generator import get_model_for_run_url

save_quantized_model = False
load_quantized_model = False
save_model_architecture = False
number_of_calibration_images = 100
dataset_path = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25"
model_wandb_url = "https://wandb.ai/gorillas/Embedding-EfficientNetRWM-CXL-OpenSet/runs/lnw2khtz/workspace"


def main() -> None:
    # 1. Quantization

    calibration_input_embeddings, _ = get_model_input(
        CXLDataset, dataset_path=dataset_path, partion="train", amount_of_tensors=number_of_calibration_images
    )

    model: Union[GraphModule, BaseModule] = get_model_for_run_url(model_wandb_url)
    if load_quantized_model:
        quantized_model_state_dict = torch.load("quantized_model_weights.pth")
        quantized_model = model
        quantized_model.load_state_dict(quantized_model_state_dict)
        quantized_model.eval()
    else:
        quantized_model = quantization_functions.ptsq_quantization_fx(model, calibration_input_embeddings)  # type: ignore

    if save_quantized_model:
        torch.save(model.state_dict(), "quantized_model_weights.pth")

    if save_model_architecture:
        log_model_to_file(quantized_model, "quantized_model.txt")
        log_model_to_file(model, "fp32_model.txt")

    # 2. Performance evaluation
    validations_input_embeddings, validation_labels = get_model_input(
        CXLDataset, dataset_path=dataset_path, partion="val", amount_of_tensors=-1
    )

    quantized_model_accuracy = performance_evaluation.get_knn_accuracy(
        model=quantized_model,
        images=validations_input_embeddings,
        labels=validation_labels,
        device=torch.device("cpu"),
        knn_number=5,
    )

    fp32_model_accuracy = performance_evaluation.get_knn_accuracy(
        model=model,
        images=validations_input_embeddings,
        labels=validation_labels,
        device=torch.device("cpu"),
        knn_number=5,
    )
    quantized_model_size = performance_evaluation.size_of_model_in_mb(quantized_model)
    fp32_model_size = performance_evaluation.size_of_model_in_mb(model)

    print(f"Quantized model accuracy: {quantized_model_accuracy}")
    print(f"FP32 model accuracy: {fp32_model_accuracy}")
    print(f"Quantized model size: {quantized_model_size} MB")
    print(f"FP32 model size: {fp32_model_size} MB")


if __name__ == "__main__":
    main()
