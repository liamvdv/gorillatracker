from typing import Tuple, Union

import torch
from torch.fx import GraphModule
import ai_edge_torch
from ai_edge_torch.quantize.quant_config import QuantConfig
from gorillatracker.model import BaseModule
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer


def convert_model_to_onnx(
    model: Union[GraphModule, BaseModule], input_shape: Tuple[int, int, int, int], output_path: str
) -> None:
    torch.onnx.export(model, torch.randn(input_shape), output_path, opset_version=17)


def convert_model_to_tflite(
    model: Union[GraphModule, BaseModule],
    input_shape: torch.Tensor,
    output_path: str,
    pt2e_quantizer: PT2EQuantizer = None,  # type: ignore
):

    pt2e_drq_model = ai_edge_torch.convert(
        model,
        (input_shape[0].unsqueeze(0),),
        quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
    )
    pt2e_drq_model.export(output_path)
