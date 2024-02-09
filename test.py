import torch
import timm

# Load the model architecture
model_architecture = "efficientnetv2_rw_m"
print(model_architecture)

# Load the weights
model_weights = torch.load("miew_id.ms_face.bin", map_location=torch.device("cpu"))

print(model_weights.keys())

# Load the model
model = timm.create_model(model_architecture, pretrained=False)

initial_state_dict = model.state_dict()

# Load the weights into the model
model.load_state_dict(model_weights, strict=False)



