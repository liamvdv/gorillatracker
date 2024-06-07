# from gorillatracker.args import TrainingArgs
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type
from urllib.parse import urlparse

import pandas as pd
import torch
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

import gorillatracker.type_helper as gtypes
from gorillatracker.data.builder import dataset_registry
from gorillatracker.data.nlet import NletDataset, build_onelet
from gorillatracker.model import BaseModule, get_model_cls

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


def get_model_for_run_url(run_url: str, eval_mode: bool = True) -> BaseModule:
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


def generate_embeddings(model: BaseModule, dataset: Any, device: str = "cpu", norm_input: bool = False) -> pd.DataFrame:
    embeddings = []
    df = pd.DataFrame(columns=["embedding", "label", "input", "label_string"])
    with torch.no_grad():
        print("Generating embeddings...")
        for ids, imgs, labels in tqdm(dataset):
            # NOTE(memben): blame me if this fails
            ids, imgs, labels = ids[0], imgs[0], labels[0]
            if isinstance(imgs, torch.Tensor):
                imgs = [imgs]
                labels = [labels]

            batch_inputs = torch.stack(imgs)
            if batch_inputs.shape[0] != 1:
                batch_inputs = batch_inputs.unsqueeze(1)
            batch_inputs = batch_inputs.to(device)

            model_inputs = batch_inputs
            if norm_input:
                model_inputs_list = [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img) for img in imgs
                ]
                model_inputs = torch.stack(model_inputs_list)
                model_inputs = model_inputs.to(device)
                if model_inputs.shape[0] != 1:
                    model_inputs = model_inputs.unsqueeze(1)
            embeddings = model(model_inputs)

            for i in range(len(imgs)):
                input_img = transforms.ToPILImage()(batch_inputs[i].cpu())
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "id": [ids],
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
    transform: Optional[gtypes.TensorTransform] = None,
) -> NletDataset:
    cls = dataset_registry[dataset_class]

    return cls(
        base_dir=Path(data_dir),
        nlet_builder=build_onelet,
        partition=partition,
        transform=model.get_tensor_transforms() if transform is None else transform,
    )


def get_latest_model_checkpoint(run: wandbRun) -> wandb.Artifact:
    models = [a for a in run.logged_artifacts() if a.type == "model"]
    return max(models, key=lambda a: a.created_at)


def generate_embeddings_from_run(
    run_url: str, outpath: str, dataset_cls: Optional[str], data_dir: Optional[str]
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

    model = get_model_for_run_url(run_url)
    args = model.config

    if data_dir is None:
        data_dir = args["data_dir"]

    if dataset_cls is None:
        dataset_cls = args["dataset_class"]

    assert data_dir is not None, "data_dir must be provided"
    assert dataset_cls is not None, "dataset_cls must be provided"

    train_dataset = get_dataset(partition="train", data_dir=data_dir, model=model, dataset_class=dataset_cls)
    val_dataset = get_dataset(partition="val", data_dir=args["data_dir"], model=model, dataset_class=dataset_cls)

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
