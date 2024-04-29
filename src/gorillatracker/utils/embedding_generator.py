# from gorillatracker.args import TrainingArgs
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Type, Union
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
from gorillatracker.scripts.create_dataset_from_videos import _crop_image
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

    if (
        "loss_module_train.prototypes" in model_state_dict or "loss_module_val.prototypes" in model_state_dict
    ):  # necessary because arcface loss also saves prototypes
        model.loss_module_train.prototypes = torch.nn.Parameter(model_state_dict["loss_module_train.prototypes"])
        model.loss_module_val.prototypes = torch.nn.Parameter(model_state_dict["loss_module_val.prototypes"])
    # note the following lines can fail if your model was not trained with the same 'embedding structure' as the current model class
    # easiest fix is to just use the old embedding structure in the model class
    model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()
    return model


def generate_embeddings(model: BaseModule, dataset: Any, device: str = "cpu", norm_input: bool = False) -> pd.DataFrame:
    embeddings = []
    df = pd.DataFrame(columns=["embedding", "label", "input", "label_string"])
    with torch.no_grad():
        print("Generating embeddings...")
        for ids, imgs, labels in tqdm(dataset):
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


def generate_embeddings_from_tracked_video(
    model: BaseModule, video_path: str, tracking_data: IdFrameDict, model_transforms: DataTransforms = lambda x: x
) -> pd.DataFrame:
    """
    Args:
        model: The model to use for embedding generation.
        video_path: Path to the video.
        tracking_data: Dictionary of Individual IDs to frames. -> {id: List[(frame_idx, (bbox))]} (bbox = (x, y, w, h)

    Returns:
        DataFrame with columns: invididual_id, frame_id, bbox, embedding,
    """
    min_frames = 15  # discard if less than 5 images
    max_per_individual = 15

    tracking_data = {
        id: frames for id, frames in tracking_data.items() if len(frames) >= min_frames
    }  # discard if less than 5 images
    print("Using", len(tracking_data), "individuals")

    video = cv2.VideoCapture(video_path)
    embedding_img_table = pd.DataFrame(columns=["embedding", "frame_id", "bbox", "invididual_id"])

    for id, frames in tracking_data.items():
        step_size = len(frames) // max_per_individual
        if step_size == 0:
            continue
        frame_list = [frames[i] for i in range(0, max_per_individual * step_size, step_size)]
        for frame_idx, bbox in frame_list:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            frame = video.read()[1]  # read the frame. read() returns a tuple of (success, frame)
            embedding = get_embedding_from_frame(model, frame, bbox, model_transforms)
            embedding_img_table = pd.concat(
                [
                    embedding_img_table,
                    pd.DataFrame(
                        {
                            "invididual_id": [id],
                            "frame_id": [frame_idx],
                            "bbox": [bbox],
                            "embedding": [embedding],
                        }
                    ),
                ],
                ignore_index=True,
            )
    video.release()
    embedding_img_table.reset_index(drop=False, inplace=True)
    return embedding_img_table


@torch.no_grad()
def get_embedding_from_frame(
    model: BaseModule, frame: cvt.MatLike, bbox: BBox, model_transforms: DataTransforms
) -> torch.Tensor:
    frame_cropped = _crop_image(
        frame,
        bbox[0],  # x
        bbox[1],  # y
        bbox[2],  # w
        bbox[3],  # h
    )

    # convert to pil image
    img = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # type: ignore
    transformed_image: torch.Tensor = model_transforms(img)

    model.eval()
    embedding = model(transformed_image.unsqueeze(0))
    return embedding


def read_embeddings_from_disk(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)
