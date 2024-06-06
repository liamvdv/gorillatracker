# from gorillatracker.args import TrainingArgs
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import cv2
import cv2.typing as cvt
import pandas as pd
import torch
import torchvision.transforms as transforms
import wandb
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from gorillatracker.model import BaseModule, get_model_cls
from gorillatracker.train_utils import get_dataset_class
from gorillatracker.type_helper import Label

DataTransforms = Union[Callable[..., Any]]
BBox = Tuple[float, float, float, float]  # x, y, w, h
BBoxFrame = Tuple[int, BBox]  # frame_idx, x, y, w, h
IdFrameDict = Dict[int, List[BBoxFrame]]  # id -> list of frames
IdDict = Dict[int, List[int]]  # id -> list of negatives
JsonDict = Dict[str, List[str]]  # video_name-id -> list of negatives
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
    wandb_fullname: str,
    model_cls: Type[BaseModule],
    model_config: Dict[str, Any],
    device: str = "cpu",
    eval_mode: bool = True,
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

    if (
        "loss_module_train.prototypes" in model_state_dict or "loss_module_val.prototypes" in model_state_dict
    ):  # necessary because arcface loss also saves prototypes
        model.loss_module_train.prototypes = torch.nn.Parameter(model_state_dict["loss_module_train.prototypes"])
        model.loss_module_val.prototypes = torch.nn.Parameter(model_state_dict["loss_module_val.prototypes"])
        # note the following lines can fail if your model was not trained with the same 'embedding structure' as the current model class
        # easiest fix is to just use the old embedding structure in the model class
    elif (
        "loss_module_train.loss.prototypes" in model_state_dict or "loss_module_val.loss.prototypes" in model_state_dict
    ):
        model.loss_module_train.loss.prototypes = torch.nn.Parameter(
            model_state_dict["loss_module_train.loss.prototypes"]
        )
        model.loss_module_val.loss.prototypes = torch.nn.Parameter(model_state_dict["loss_module_val.loss.prototypes"])
    model.load_state_dict(model_state_dict)

    model.to(device)
    if eval_mode:
        model.eval()
    return model


def get_model_from_run(run, eval_mode: bool = True) -> BaseModule:
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
            "stepwise_schedule",
            "lr_interval",
            "l2_alpha",
            "l2_beta",
            "path_to_pretrained_weights",
            # NOTE(liamvdv): might need be extended by other keys if model keys change
        )
    }

    print("Loading model from latest checkpoint")
    model_path = get_latest_model_checkpoint(run).qualified_name
    model_cls = get_model_cls(args["model_name_or_path"])
    return load_model_from_wandb(model_path, model_cls=model_cls, model_config=args, eval_mode=eval_mode)


def generate_embeddings(model: BaseModule, dataset: Any, device: str = "cuda", batch_size: int = 256, worker: int = 1) -> pd.DataFrame:
    model = model.to(device)
    all_ids = []
    all_embeddings = []
    all_labels = []
    all_inputs = []
    all_label_strings = []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=worker, shuffle=False)
    
    with torch.no_grad():
        print("Generating embeddings...")
        for batch in tqdm(data_loader):
            ids, imgs, labels = batch
            batch_inputs = imgs.to(device)

            embeddings = model(batch_inputs).cpu()
            for i in range(batch_inputs.size(0)):
                input_img = transforms.ToPILImage()(batch_inputs[i].cpu())
                label_string = dataset.mapping[labels[i].item()] if hasattr(dataset, "mapping") else None
                
                all_ids.append(ids[i].item())
                all_embeddings.append(embeddings[i].numpy())
                all_labels.append(labels[i].item())
                all_inputs.append(input_img)
                all_label_strings.append(label_string)
                
    df = pd.DataFrame({
        "id": all_ids,
        "embedding": all_embeddings,
        "label": all_labels,
        "input": all_inputs,
        "label_string": all_label_strings
    })
    
    return df

# def generate_embeddings(model: BaseModule, dataset: Any, device: str = "cuda", batch_size: int = 256) -> pd.DataFrame:
#     model = model.to(device)
#     embeddings = []
#     df = pd.DataFrame(columns=["embedding", "label", "input", "label_string"])
#     with torch.no_grad():
#         print("Generating embeddings...")
#         for ids, imgs, labels in tqdm(dataset):
#             if isinstance(imgs, torch.Tensor):
#                 imgs = [imgs]
#                 labels = [labels]

#             batch_inputs = torch.stack(imgs)
#             if batch_inputs.shape[0] != 1:
#                 batch_inputs = batch_inputs.unsqueeze(1)
#             batch_inputs = batch_inputs.to(device)

#             embeddings = model(batch_inputs).cpu()
#             for i in range(len(imgs)):
#                 input_img = transforms.ToPILImage()(batch_inputs[i].cpu())
#                 df = pd.concat(
#                     [
#                         df,
#                         pd.DataFrame(
#                             {
#                                 "id": [ids],
#                                 "embedding": [embeddings[i]],
#                                 "label": [labels[i]],
#                                 "input": [input_img],
#                                 "label_string": [dataset.mapping[labels[i]]] if hasattr(dataset, "mapping") else None,
#                             }
#                         ),
#                     ]
#                 )
#     df.reset_index(drop=False, inplace=True)
#     return df


def get_dataset(
    model: BaseModule,
    partition: Literal["train", "val", "test"],
    data_dir: str,
    dataset_class: str,
    transform: Union[Callable[..., Any], None] = None,
) -> Dataset[Tuple[Any, Label]]:
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


def generate_embeddings_from_run(
    run_url: str, outpath: str, dataset_cls: Optional[str] = None, data_dir: Optional[str] = None
) -> pd.DataFrame:
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
    model = get_model_from_run(run)

    if data_dir is None:
        print("Using data_dir from run")
        data_dir = run.config["data_dir"]

    if dataset_cls is None:
        print("Using dataset_class from run")
        dataset_cls = run.config["dataset_class"]

    train_dataset = get_dataset(partition="train", data_dir=data_dir, model=model, dataset_class=dataset_cls)
    val_dataset = get_dataset(partition="val", data_dir=data_dir, model=model, dataset_class=dataset_cls)

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
