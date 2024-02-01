from typing import Any, Callable

import torch
import torchvision.transforms as transforms
import wandb
import pandas as pd
from tqdm import tqdm

from gorillatracker.datasets.cxl import CXLDataset
from gorillatracker.model import BaseModule


# load model from wandb artifact
def load_model_from_wandb(wandb_fullname, model_cls: Any, embedding_size, device: str) -> Any:  # TODO
    wandb.login()
    wandb.init(mode="disabled")
    api = wandb.Api()

    artifact = api.artifact(
        wandb_fullname,  # your artifact name
        type="model",
    )
    artifact_dir = artifact.download()
    model = artifact_dir + "/model.ckpt"
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["state_dict"]

    model = model_cls(
        embedding_size=embedding_size,
    )

    if not hasattr(model_state_dict, "loss_module_train.prototypes") or not hasattr(
        model_state_dict, "loss_module_val.prototypes"
    ):  # necessary because arcface loss also saves prototypes
        model.loss_module_train.prototypes = torch.nn.Parameter(model_state_dict["loss_module_train.prototypes"])
        model.loss_module_val.prototypes = torch.nn.Parameter(model_state_dict["loss_module_val.prototypes"])

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model


# generate embeddings (dataframe) from model and dataset
def generate_embeddings(model, dataset, device):  # TODO
    embeddings = []
    df = pd.DataFrame(columns=["embedding", "label", "input", "label_string"])
    with torch.no_grad():
        print("Generating embeddings...")
        for imgs, labels in tqdm(dataset):
            if isinstance(imgs, torch.Tensor):  # if single image is passe wrap it in list
                imgs = [imgs]
                labels = [labels]
            batch_inputs = torch.stack(imgs)
            if batch_inputs.shape[0] != 1:
                batch_inputs = batch_inputs.unsqueeze(1)
            batch_inputs = batch_inputs.to(device)
            embeddings = model(batch_inputs)

            for i in range(len(imgs)):
                input_img = transforms.ToPILImage()(batch_inputs[i].cpu())
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "embedding": [embeddings[i]],
                                "label": [labels[i]],
                                "input": [input_img],
                                "label_string": [dataset.mapping[labels[i]]],
                            }
                        ),
                    ]
                )
    df.reset_index(drop=False, inplace=True)
    return df


def get_dataset(
    partition: str = "val",
    data_dir: str = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
    transform: Callable = None,
    model: BaseModule = BaseModule(),
):
    if transform is None:
        transform = transforms.Compose(
            [
                CXLDataset.get_transforms(),
                model.get_tensor_transforms(),
            ]
        )

    return CXLDataset(
        data_dir=data_dir,
        partition=partition,
        transform=transform,
    )
