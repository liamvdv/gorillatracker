# from gorillatracker.args import TrainingArgs
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Tuple, Type, Union
from urllib.parse import urlparse

import pandas as pd
import torch
import torchvision.transforms as transforms
import wandb
from torch.utils.data import Dataset
from tqdm import tqdm

from gorillatracker.model import BaseModule, get_model_cls
from gorillatracker.train_utils import get_dataset_class
from gorillatracker.type_helper import Image, Label

wandbRun = Any


def get_wandb_api() -> wandb.Api:
    if not hasattr(get_wandb_api, "api"):
        get_wandb_api.api = wandb.Api()  # type: ignore
    return get_wandb_api.api  # type: ignore


def parse_wandb_url(url: str) -> Tuple[str, str, str]:
    assert url.startswith("https://wandb.ai/")
    parsed = urlparse(url)
    assert parsed.netloc == "wandb.ai"
    print(parsed, parsed.path.split("/"), parsed.path)
    parts = parsed.path.strip("/").split(
        "/"
    )  # ['gorillas', 'Embedding-SwinV2-CXL-Open', 'runs', 'fnyvl65k', 'overview']
    entity, project, s_runs, run_id, *rest = parts
    assert (
        s_runs == "runs"
    ), "expect: https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/fnyvl65k/overview like format."
    return entity, project, run_id


def get_run(url: str) -> wandbRun:
    # https://docs.wandb.ai/ref/python/run
    entity, project, run_id = parse_wandb_url(url)
    run = get_wandb_api().run(f"{entity}/{project}/{run_id}")  # type: ignore
    return run


def load_model_from_wandb(
    wandb_fullname: str, model_cls: Type[BaseModule], model_config: Dict[str, Any], device: str = "cpu"
) -> BaseModule:
    api = get_wandb_api()

    artifact = api.artifact(  # type: ignore
        wandb_fullname,
        type="model",
    )
    artifact_dir = artifact.download()
    model = artifact_dir + "/model.ckpt"  # all of our models are saved as model.ckpt
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["state_dict"]

    model = model_cls(**model_config)

    if hasattr(model_state_dict, "loss_module_train.prototypes") and hasattr(
        model_state_dict, "loss_module_val.prototypes"
    ):  # necessary because arcface loss also saves prototypes
        model.loss_module_train.prototypes = torch.nn.Parameter(model_state_dict["loss_module_train.prototypes"])
        model.loss_module_val.prototypes = torch.nn.Parameter(model_state_dict["loss_module_val.prototypes"])

    # note the following lines can fail if your model was not trained with the same 'embedding structure' as the current model class
    # easiest fix is to just use the old embedding structure in the model class
    model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()
    return model


def generate_embeddings(model: BaseModule, dataset: Any, device: str = "cpu") -> pd.DataFrame:
    embeddings = []
    df = pd.DataFrame(columns=["embedding", "label", "input", "label_string"])
    with torch.no_grad():
        print("Generating embeddings...")
        for imgs, labels in tqdm(dataset):
            if isinstance(imgs, torch.Tensor):
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
                                "label_string": [dataset.mapping[labels[i]]] if dataset.mapping else None,
                            }
                        ),
                    ]
                )
    df.reset_index(drop=False, inplace=True)
    return df


def get_dataset(
    model: BaseModule,
    partition: Literal["train", "val", "test"],
    data_dir: str,
    dataset_class: str,
    transform: Union[Callable[..., Any], None] = None,
) -> Dataset[Tuple[Image, Label]]:
    cls = get_dataset_class(dataset_class)
    if transform is None:
        transform = transforms.Compose(
            [
                cls.get_transforms(),  # type: ignore
                model.get_tensor_transforms(),
            ]
        )

    return cls(  # type: ignore
        data_dir=data_dir,
        partition=partition,
        transform=transform,
    )


def get_latest_model_checkpoint(run: wandbRun) -> wandb.Artifact:
    models = [a for a in run.logged_artifacts() if a.type == "model"]
    return max(models, key=lambda a: a.created_at)


def generate_embeddings_from_run(run_url: str, outpath: str) -> pd.DataFrame:
    """
    generate a pandas df that generates embeddings for all images in the dataset partitions train and val.
    stores to DataFrame
    partition, image_path, embedding, label, label_string
    """
    out = Path(outpath)
    is_write = outpath != "-"
    if is_write:
        assert not out.exists(), "outpath must not exist"
        assert out.parent.exists(), "outpath parent must exist"
        assert out.suffix == ".pkl", "outpath must be a pickle file"

    run = get_run(run_url)
    print("Using model from run:", run.name)
    print("Config:", run.config)
    # args = TrainingArgs(**run.config) # NOTE(liamvdv): contains potenially unknown keys / missing keys (e. g. l2_beta)
    args = {
        k: run.config[k]
        for k in (
            # Others:
            "model_name_or_path",
            "dataset_class",
            "data_dir",
            # Model Params:
            "embedding_size",
            "from_scratch",
            "loss_mode",
            "weight_decay",
            "lr_schedule",
            "warmup_mode",
            "warmup_epochs",
            "max_epochs",
            "initial_lr",
            "start_lr",
            "end_lr",
            "beta1",
            "beta2",
            # NOTE(liamvdv): might need be extended by other keys if model keys change
        )
    }

    print("Loading model from latest checkpoint")
    model_path = get_latest_model_checkpoint(run).qualified_name
    model_cls = get_model_cls(args["model_name_or_path"])
    model = load_model_from_wandb(model_path, model_cls=model_cls, model_config=args)

    train_dataset = get_dataset(
        partition="train", data_dir=args["data_dir"], model=model, dataset_class=args["dataset_class"]
    )
    val_dataset = get_dataset(
        partition="val", data_dir=args["data_dir"], model=model, dataset_class=args["dataset_class"]
    )

    val_df = generate_embeddings(model, val_dataset)
    val_df["partition"] = "val"

    train_df = generate_embeddings(model, train_dataset)
    train_df["partition"] = "train"

    df = pd.concat([train_df, val_df], ignore_index=True)

    print("Embeddings for", len(df), "images generated")

    # store
    if is_write:
        df.to_pickle(outpath)
    print("done")
    return df


def read_embeddings_from_disk(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)
