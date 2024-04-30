import torch
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn


def calibrate(model: nn.Module, calibration_input: torch.Tensor):
    model.eval()
    with torch.no_grad():
        model(calibration_input)


def dynamic_default_quantization(model, dtype=torch.qint8):
    """This function only supports quantizing the following layer types:
    - nn.LSTM
    - nn.Linear
    """
    return torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=dtype,
    )


def ptsq_quantization(model, calibration_input) -> nn.Module:
    """https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization"""
    model = model.model
    model.qconfig = torch.quantization.get_default_qconfig("x86")

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32_fused = torch.quantization.fuse_modules(model, [["conv", "batchnorm"]])

    model_fp32_fused = model
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    calibrate(model_fp32_prepared, calibration_input)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    return model_int8


def ptsq_quantization_fx(model, calibration_input) -> nn.Module:
    """https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization"""

    model = model.model
    qconfig = torch.ao.quantization.get_default_qconfig_mapping("fbgemm")
    model_fp32_fused = model
    model_fp32_prepared = quantize_fx.prepare_fx(model_fp32_fused, qconfig, calibration_input)
    calibrate(model_fp32_prepared, calibration_input)
    model_fp32_prepared(calibration_input)
    model_int8 = quantize_fx.convert_fx(model_fp32_prepared)
    return model_int8


# def pt2e_quantization(model, dtype=torch.qint8):
#     """https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html"""
#     m = capture_pre_autograd_graph(model, *example_inputs)

#     quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
#     m = prepare_pt2e(m, quantizer)
#     m = convert_pt2e(m)
#     return m
