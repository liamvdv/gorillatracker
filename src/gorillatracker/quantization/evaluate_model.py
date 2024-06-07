import argparse
import os
from typing import Any, Dict, Union

import pandas as pd
import torch
import yaml
from print_on_steroids import print_on_steroids
from torch.fx import GraphModule

import gorillatracker.quantization.quantization_functions as quantization_functions
from gorillatracker.datasets.cxl import CXLDataset
from gorillatracker.model import BaseModule
from gorillatracker.quantization.export_model import convert_model_to_tflite
from gorillatracker.quantization.performance_evaluation import evaluate_model
from gorillatracker.quantization.utils import get_model_input, log_model_to_file
from gorillatracker.utils.embedding_generator import get_model_for_run_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation")

    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--base_dir", type=str, default="runs/", help="Base directory for saving runs")
    parser.add_argument("--save_quantized_model", action="store_true", help="Flag to save the quantized model")
    parser.add_argument("--load_quantized_model", action="store_true", help="Flag to load the quantized model")
    parser.add_argument("--save_model_architecture", action="store_true", help="Flag to save the model architecture")
    parser.add_argument("--number_of_calibration_images", type=int, default=100, help="Number of calibration images")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
    )
    parser.add_argument("--model_wandb_url", type=str, help="WandB URL for the model")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file", default="configs/config.yaml")

    args = parser.parse_args()

    if args.config_path:
        opt = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
        args_list = opt
        for key in args_list:
            setattr(args, key, args_list[key])

    return args


def main(args) -> None:  # type: ignore
    base_dir = args.base_dir + "/" + args.name
    # 1. Quantization
    calibration_input_embeddings, _ = get_model_input(
        CXLDataset, dataset_path=args.dataset_path, partion="train", amount_of_tensors=args.number_of_calibration_images
    )

    model: BaseModule = get_model_for_run_url(args.model_wandb_url)
    if args.load_quantized_model:
        quantized_model_state_dict = torch.load("quantized_model_weights.pth")
        quantized_model: Union[GraphModule, BaseModule] = model
        quantized_model.load_state_dict(quantized_model_state_dict)
        quantized_model.eval()
    else:
        quantized_model, quantizer = quantization_functions.pt2e_quantization(model, calibration_input_embeddings)

    if args.save_quantized_model:
        exported_model = torch.export.export(quantized_model, (calibration_input_embeddings[0].unsqueeze(0),))
        torch.save(exported_model, os.path.join(base_dir, "quantized_model_weights.pth"))

    if args.save_model_architecture:
        log_model_to_file(quantized_model, os.path.join(base_dir, "quantized_model.txt"))
        log_model_to_file(model, os.path.join(base_dir, "fp32_model.txt"))

    print_on_steroids("Quantization done", level="success")

    # 2. Performance evaluation
    validations_input_embeddings, validation_labels = get_model_input(
        CXLDataset, dataset_path=args.dataset_path, partion="val", amount_of_tensors=-1
    )

    results: Dict[str, Any] = dict()

    evaluate_model(model, "fp32", results, validations_input_embeddings, validation_labels)
    evaluate_model(quantized_model, "quantized", results, validations_input_embeddings, validation_labels)

    print_on_steroids("Model evaluation done", level="success")

    # 3. Export to TF Lite
    tf_model = convert_model_to_tflite(
        quantized_model, calibration_input_embeddings[0], os.path.join(base_dir, "quantized_model.tflite")
    )

    evaluate_model(tf_model, "tflite", results, validations_input_embeddings, validation_labels)

    pd.DataFrame(results).to_json(os.path.join(base_dir, "results.json"))
    print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)
