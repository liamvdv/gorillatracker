from typing import Literal, Type

import torch
import torch.nn as nn
from torchvision import transforms

from gorillatracker.datasets.cxl import CXLDataset
from gorillatracker.transform_utils import SquarePad


def get_model_input(
    dataset_cls: Type[CXLDataset],
    dataset_path: str,
    partion: Literal["train", "val", "test"] = "train",
    amount_of_tensors: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a tensor of images and a tensor of labels from a dataset.
    Args:
        dataset_cls: The dataset class to use.
        dataset_path: The path to the dataset.
        partion: The partion of the dataset to use.
        amount_of_tensors: The amount of tensors to get from the dataset. If -1, get all tensors.
    """

    dataset = dataset_cls(
        dataset_path,
        partion,
        transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )

    if amount_of_tensors == -1:
        amount_of_tensors = len(dataset)

    images = []
    labels = []
    for i in range(amount_of_tensors):
        _, image, label = dataset[i]
        images.append(image)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels)


def log_model_to_file(model: nn.Module, file_name: str = "model.txt") -> None:
    with open(file_name, "w") as f:
        for name, layer in model.named_modules():
            if len(list(layer.parameters())) > 0:  # Ensure the layer has parameters
                precisions = [param.dtype for param in layer.parameters()]
                f.write(f"Layer: {name}\n{layer}\nPrecisions: {precisions}\n\n")
            else:
                f.write(f"Layer: {name}\n{layer}\nPrecisions: None (No parameters)\n\n")


def print_model_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        print(f"{name}: {param.nelement()} parameters")


def load_quantized_model(model: nn.Module, model_path: str) -> nn.Module:
    """Load a quantized model from a file.
    Args:
        model: The model to load the parameters into.
        model_path: The path to the model file.
    """
    model.load_state_dict(torch.load(model_path))
    return model
