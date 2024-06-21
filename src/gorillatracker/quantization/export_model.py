from typing import Tuple, Union

import ai_edge_torch
import torch
from ai_edge_torch.model import TfLiteModel
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.quantize.quant_config import QuantConfig
from print_on_steroids import print_on_steroids
from torch.fx import GraphModule

from gorillatracker.model import BaseModule


def convert_model_to_onnx(
    model: Union[GraphModule, BaseModule], input_shape: Tuple[int, int, int, int], output_path: str
) -> None:
    torch.onnx.export(model, torch.randn(input_shape), output_path, opset_version=17)


def convert_model_to_tflite(
    model: Union[GraphModule, BaseModule],
    input_shape: torch.Tensor,
    output_path: str,
    pt2e_quantizer: PT2EQuantizer = None,
) -> TfLiteModel:
    print_on_steroids("Conversion to tflite", level="info")
    pt2e_drq_model = ai_edge_torch.convert(
        model,
        (input_shape[0].unsqueeze(0),),
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
    )
    pt2e_drq_model.export(output_path)
    print_on_steroids("Conversion to tflite done.", level="success")
    return pt2e_drq_model
