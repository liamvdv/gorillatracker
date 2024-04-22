import os
from typing import Any, Dict, Union

import yaml
from wandb import agent, sweep


def check_sweep_configs(configs: list[dict[str, Any]]) -> None:
    for config in configs:
        assert (
            len(config["project_name"].split("-")) >= 4
        ), "Project name must be of the form <Function>-<Backbone>-<Dataset>-<Set-Type>"
        get_config(config["config_path"])


def get_config(config_path: str) -> Dict[str, Any]:
    assert os.path.isfile(config_path), f"Config file not found at {config_path}"
    assert config_path.endswith(".yml") or config_path.endswith(".yaml"), "Config file must be YAML"
    config_dict: Dict[str, Union[str, int, float, bool]] = dict()
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def run_sweep(project_name: str, entity: str, config_path: str, parameters: Dict[str, Dict[str, Any]]) -> None:
    sweep_config = {
        "program": "./train.py",  # Note: not the sweep file, but the training script
        "name": project_name,
        "method": "grid",  # Specify the search method (random search in this case)
        "metric": {"goal": "maximize", "name": "val/embeddings/knn/accuracy"},  # Specify the metric to optimize
        "parameters": parameters,
        "command": ["${interpreter}", "${program}", "${args}", "--config_path", config_path],
    }
    sweep_id = sweep(sweep=sweep_config, project=project_name, entity=entity)
    # Print the sweep ID directly
    print(f"SWEEP_PATH={entity}/{project_name}/{sweep_id}")
    agent(sweep_id)


sweeps = [
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "32-true"},
            "wandb_tags": {"value": ["fp32-true"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "16-true"},
            "wandb_tags": {"value": ["fp16-true"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "bf16-true"},
            "wandb_tags": {"value": ["bf16-true"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "bf16-mixed"},
            "wandb_tags": {"value": ["bf16-mixed"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "transformer-engine-float16"},
            "wandb_tags": {"value": ["fp8", "weights-bf16"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "transformer-engine"},
            "wandb_tags": {"value": ["fp8", "weights-fp16"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "int8-training"},
            "wandb_tags": {"value": ["int8-activations", "weights-fp16", "fp16-computations"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "int8"},
            "wandb_tags": {"value": ["int8", "fp16-computations"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "fp4"},
            "wandb_tags": {"value": ["fp4", "fp16-computations"]},
        },
    },
    {
        "project_name": "Embedding-EfficientNet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "precision": {"value": "nf4"},
            "wandb_tags": {"value": ["nf4", "fp16-computations"]},
        },
    },
]

check_sweep_configs(sweeps)

for current_sweep in sweeps:
    print(f"Running sweep: {current_sweep['project_name']}")
    try:
        run_sweep(**current_sweep)  # type: ignore
    except Exception as e:
        print(f"Error running sweep: {current_sweep['project_name']}")
        print(e)
        continue
