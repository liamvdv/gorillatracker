from typing import Tuple

import torch

from gorillatracker.model import BaseModule


def convert_model_to_onnx(model: BaseModule, input_shape: Tuple[int, int, int, int], output_path: str) -> None:
    torch.onnx.export(model, torch.randn(input_shape), output_path, opset_version=17)
