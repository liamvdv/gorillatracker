from typing import Tuple

import torch
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import allow_exported_model_train_eval  # type: ignore
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config
from torch.fx import GraphModule

from gorillatracker.model import BaseModule


def calibrate(model: nn.Module, calibration_input: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        model(calibration_input)


def dynamic_default_quantization(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """This function only supports quantizing the following layer types:
    - nn.LSTM
    - nn.Linear
    """
    return torch.quantization.quantize_dynamic(  # type: ignore
        model,
        {nn.LSTM, nn.Linear},
        dtype=dtype,
    )


def ptsq_quantization(model: BaseModule, calibration_input: torch.Tensor) -> Tuple[GraphModule, None]:
    """https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization"""
    model = model.model
    model.qconfig = torch.quantization.get_default_qconfig("x86")  # type: ignore

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32_fused = torch.quantization.fuse_modules(model, [["conv", "batchnorm"]])

    model_fp32_fused = model
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)  # type: ignore
    calibrate(model_fp32_prepared, calibration_input)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)  # type: ignore
    return model_int8, None


def ptsq_quantization_fx(model: BaseModule, calibration_input: torch.Tensor) -> Tuple[GraphModule, None]:
    """https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization"""

    model = model.model
    qconfig = torch.ao.quantization.get_default_qconfig_mapping("fbgemm")
    model_fp32_fused = model
    model_fp32_prepared = quantize_fx.prepare_fx(model_fp32_fused, qconfig, calibration_input)  # type: ignore
    calibrate(model_fp32_prepared, calibration_input)
    model_fp32_prepared(calibration_input)
    model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
    return model_int8, None


def pt2e_quantization(model: BaseModule, calibration_input: torch.Tensor) -> Tuple[GraphModule, PT2EQuantizer]:
    """https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html"""
    calibration_input_tuple = (calibration_input,)
    prepared_model_autograd = capture_pre_autograd_graph(model, calibration_input_tuple)

    quantizer = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
    prepared_model = prepare_pt2e(prepared_model_autograd, quantizer)
    allow_exported_model_train_eval(prepared_model)
    calibrate(prepared_model, calibration_input)

    quantized_model = convert_pt2e(prepared_model, fold_quantize=False)
    allow_exported_model_train_eval(quantized_model)
    return quantized_model, quantizer
