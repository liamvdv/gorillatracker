from typing import Type
from urllib.parse import urlparse

import wandb
from wandb.apis.public.runs import Run

from gorillatracker.model import BaseModule, get_model_cls


def parse_wandb_url(url: str) -> tuple[str, str, str]:
    assert url.startswith("https://wandb.ai/")
    parsed = urlparse(url)
    assert parsed.netloc == "wandb.ai"
    parts = parsed.path.strip("/").split(
        "/"
    )  # ['gorillas', 'Embedding-SwinV2-CXL-Open', 'runs', 'fnyvl65k', 'overview']
    entity, project, s_runs, run_id, *_ = parts
    assert (
        s_runs == "runs"
    ), "expect: https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/fnyvl65k/overview like format."
    return entity, project, run_id


def get_run(url: str) -> Run:
    # https://docs.wandb.ai/ref/python/run
    entity, project, run_id = parse_wandb_url(url)
    wandb_api = wandb.Api()
    run = wandb_api.run(path=f"{entity}/{project}/{run_id}")  # type: ignore[no-untyped-call]
    return run


def get_latest_model_checkpoint(run: Run) -> wandb.Artifact:
    models: list[wandb.Artifact] = [a for a in run.logged_artifacts() if a.type == "model"]  # type: ignore[no-untyped-call]
    return max(models, key=lambda a: a.created_at)


def load_model(model_cls: Type[BaseModule], model_path: str) -> BaseModule:
    print(model_cls)
    print(model_path)
    model = model_cls.load_from_checkpoint(model_path, data_module=None, wandb_run=None)
    return model


def get_model_for_run_url(run_url: str) -> BaseModule:
    run = get_run(run_url)
    model_cls = get_model_cls(run.config["model_name_or_path"])
    artifact = get_latest_model_checkpoint(run)
    artifact_dir = artifact.download()
    model = artifact_dir + "/model.ckpt"
    return load_model(model_cls, model)
