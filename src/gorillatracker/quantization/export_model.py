import torch

# import onnx
# import torchvision

model_path = "quantized_model.pth"
model = torch.load(model_path)


# 1. Convert the model to ONNX
input_shape = (1, 3, 224, 224)
torch.onnx.export(model, torch.randn(input_shape), "quantized_model.onnx", opset_version=17)

# 2. Convert ONNX to TensorFlow

# 3. Convert TensorFlow to TensorFlow Lite

# 4. Export the TensorFlow Lite model to Edge TPU
