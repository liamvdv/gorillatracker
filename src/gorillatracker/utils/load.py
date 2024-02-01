from typing import Any

import torch
import wandb
import pandas as pd
from tqdm import tqdm


# load model from wandb artifact
def load_model_from_artifact(artifact: wandb.Artifact, model: Any, device: str) -> Any: # TODO
    model_state_dict = artifact.get('model_state_dict').state_dict()
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model

# generate embeddings (dataframe) from model and dataset
def generate_embeddings(model, dataset, device): # TODO
    model.eval()
    embeddings = []
    df = pd.DataFrame(columns=['embedding', 'label', 'input', 'label_string'])
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dataset): # TODO
            batch_inputs = batch_inputs.to(device)
            embeddings = model(batch_inputs)
            for i in range(len(batch_inputs)):
                df.append({'embedding': embeddings[i], 'label': batch_labels[i], 'input': batch_inputs[i], 'label_string': dataset.mapping[batch_labels[i]]}, ignore_index=True, inplace=True)
    df.reset_index(drop=False, inplace=True)
    return df

""" Code from my notebook

wandb.login()
wandb.init(mode="disabled")
api = wandb.Api()

artifact = api.artifact(
    "gorillas/Embedding-ALL-SPAC-Open/model-3ag1c2vf:v1",  # your artifact name
    type="model",
)
artifact_dir = artifact.download()
model = artifact_dir + "/model.ckpt"

# load model
checkpoint = torch.load(model, map_location=torch.device("cpu"))

model = EfficientNetV2Wrapper(  # switch this with the model you want to use
    model_name_or_path="EfficientNetV2_Large",
    from_scratch=False,
    loss_mode="softmax/arcface",
    weight_decay=0.001,
    lr_schedule="cosine",
    warmup_mode="cosine",
    warmup_epochs=10,
    max_epochs=100,
    initial_lr=0.01,
    start_lr=0.01,
    end_lr=0.0001,
    beta1=0.9,
    beta2=0.999,
    embedding_size=128,
)
# the following lines are necessary to load a model that was trained with arcface (the prototypes are saved in the state dict)
model.loss_module_train.prototypes = torch.nn.Parameter(checkpoint["state_dict"]["loss_module_train.prototypes"])
model.loss_module_val.prototypes = torch.nn.Parameter(checkpoint["state_dict"]["loss_module_val.prototypes"])

model.load_state_dict(checkpoint["state_dict"])
model.eval()

# generate table that contains labels and images and embeddings
df = pd.DataFrame(columns=["label", "image", "embedding"])
dataset = CXLDataset(
    data_dir="/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
    partition="val",
    transform=transforms.Compose(  # use the transforms that were used for the model (except of course data augmentations)
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    ),
)

for i in range(len(dataset)):
    image_tensor, label = dataset[i]
    label_string = dataset.mapping[label]
    image = transforms.ToPILImage()(image_tensor)
    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])(
        image_tensor
    )  # if your model was trained with normalization, you need to normalize the images here as well
    embedding = model(image_tensor.unsqueeze(0))
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "label_string": [label_string],
                    "label": [label],
                    "image": [image],
                    "embedding": [embedding[0].detach().numpy()],
                }
            ),
        ]
    )

    if i % 10 == 0:
        print(f"\rprocessed {i} images")
df = df.reset_index(drop=False)
"""