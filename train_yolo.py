import shutil
import subprocess
import time
from dataclasses import dataclass

import typer
from ultralytics import YOLO

app = typer.Typer()


@dataclass(frozen=True)
class BaseConfig:
    project: str
    base_model = "/workspaces/gorillatracker/yolov8n.pt"
    training_name: str
    data: str
    single_cls: bool = False


def get_dataset_config(dataset_key: str, training_name: str) -> BaseConfig:
    datasets = {
        "fmcb": BaseConfig(
            project="Detection-YOLOv8-CXLBodyFemaleMaleChild-ClosedSet",
            data="/workspaces/gorillatracker/cfgs/yolo_detection_body_fmc.yaml",
            training_name=training_name,
        ),
        "gsb": BaseConfig(
            project="Detection-YOLOv8-CXLBodyGorillaSilverback-ClosedSet",
            data="/workspaces/gorillatracker/cfgs/yolo_detection_body_gs.yaml",
            training_name=training_name,
        ),
        "gb": BaseConfig(
            project="Detection-YOLOv8-CXLBodyGorilla-ClosedSet",
            data="/workspaces/gorillatracker/cfgs/yolo_detection_body_gs.yaml",
            training_name=training_name,
            single_cls=True,
        ),
        "gf90": BaseConfig(
            project="Detection-YOLOv8-CXLFace90DegreeGorilla-ClosedSet",
            data="/workspaces/gorillatracker/cfgs/yolo_detection_face_90.yaml",
            training_name=training_name,
        ),
        "gf45": BaseConfig(
            project="Detection-YOLOv8-CXLFace45DegreeGorilla-ClosedSet",
            data="/workspaces/gorillatracker/cfgs/yolo_detection_face_45.yaml",
            training_name=training_name,
        ),
    }
    assert dataset_key in datasets
    return datasets[dataset_key]


@app.command()
def tune(
    dataset_key: str,
    training_name: str,
    iterations: int = 20,
    epochs: int = 200,
) -> None:
    config = get_dataset_config(dataset_key, training_name)
    model = YOLO(config.base_model)
    model.tune(
        project=config.project,
        data=config.data,
        name=config.training_name,
        iterations=iterations,
        epochs=epochs,
    )
    shutil.move(config.project, f"logs/{config.project}-{config.training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


@app.command()
def train(
    dataset_key: str,
    training_name: str,
    epochs: int = 200,
    batch_size: int = -1,  # -1 for autobatch
    patience: int = 40,
) -> None:
    config = get_dataset_config(dataset_key, training_name)
    model = YOLO(config.base_model)
    model.train(
        project=config.project,
        name=config.training_name,
        data=config.data,
        epochs=epochs,
        batch=batch_size,
        patience=patience,
        single_cls=config.single_cls,
    )
    shutil.move(config.project, f"logs/{config.project}-{config.training_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}")


@app.command()
def tune_batch_size_all(
    training_name: str,
    batch_sizes: str = "8,16,32,64,-1",  # -1 for autobatch
    epochs: int = 200,
    patience: int = 40,
) -> None:
    sizes = [int(x) for x in batch_sizes.split(",")]
    for dataset_key in ["fmcb", "gsb", "gb", "gf90", "gf45"]:
        for bs in sizes:
            subprocess.run(
                [
                    "python",
                    "train_yolo.py",
                    "train",
                    dataset_key,
                    f"{training_name}-{dataset_key}",
                    "--epochs",
                    str(epochs),
                    "--patience",
                    str(patience),
                    "--batch-size",
                    str(bs),
                ]
            )


@app.command()
def tune_all(
    training_name: str,
    iterations: int = 20,
    epochs: int = 200,
) -> None:
    # this is (unfortunately) needed to solve WANDB issues
    for dataset_key in ["fmcb", "gsb", "gb", "gf90", "gf45"]:
        subprocess.run(
            [
                "python",
                "train_yolo.py",
                "tune",
                dataset_key,
                f"{training_name}-{dataset_key}",
                "--iterations",
                str(iterations),
                "--epochs",
                str(epochs),
            ]
        )


if __name__ == "__main__":
    app()
