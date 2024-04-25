import torch.quantization
import os
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import transforms
from gorillatracker.datasets.cxl import CXLDataset
from torch._export import capture_pre_autograd_graph

# from torch.ao.quantization.quantizer import (
#     XNNPACKQuantizer,
#     get_symmetric_quantization_config,
# )

from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)

from gorillatracker.utils.embedding_generator import get_model_for_run_url


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


def ptsq_quantization(model, calibration_input, dtype=torch.qint8):
    """https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization"""
    model = model.model
    model.qconfig = torch.quantization.get_default_qconfig("x86")
    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    # model_fp32_fused = torch.quantization.fuse_modules(model, [["conv", "batchnorm"]])
    model_fp32_fused = model

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    model_fp32_prepared(calibration_input)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    return model_int8


def pt2e_quantization(model, dtype=torch.qint8):
    """https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html"""
    m = capture_pre_autograd_graph(model, *example_inputs)

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    m = prepare_pt2e(m, quantizer)
    m = convert_pt2e(m)
    return m


def log_model_to_file(model, file_name):
    with open("model.txt", "w") as f:
        f.write(str(model))


def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.nelement()} parameters")


def size_of_model_in_mb(model):
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return model_size


def get_calibration_input(
    dataset_cls, dataset_path: str, partion: str = "train", calibration_size: int = 100
) -> list[torch.Tensor]:
    dataset = dataset_cls(
        dataset_path,
        partion,
        transforms.Compose(
            [
                transforms.ToTensor(),
                dataset_cls.get_transforms(),
            ]
        ),
    )

    return [element[0] for element in dataset[:calibration_size]]


model = get_model_for_run_url("https://wandb.ai/gorillas/Embedding-EfficientNet-CXL-OpenSet/runs/m9kacnwe")


calibration_input = get_calibration_input(
    CXLDataset,
    "/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
)
quantized_model = ptsq_quantization(model, calibration_input)

print("Model size before quantization: ", size_of_model_in_mb(model), "MB")
print("Model size after quantization: ", size_of_model_in_mb(quantized_model), "MB")
