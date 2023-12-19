import os
from typing import Dict, Union

import yaml

from wandb import agent, sweep


def check_sweep_configs(configs: Dict[str, Dict]):
    for config in configs:
        assert (
            len(config["project_name"].split("-")) >= 4
        ), "Project name must be of the form <Function>-<Backbone>-<Dataset>-<Set-Type>"
        get_config(config["config_path"])


def get_config(config_path: str) -> Dict:
    assert os.path.isfile(config_path), f"Config file not found at {config_path}"
    assert config_path.endswith(".yml") or config_path.endswith(".yaml"), "Config file must be YAML"
    config_dict: Dict[str, Union[str, int, float, bool]] = dict()
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def run_sweep(project_name: str, entity: str, config_path: str, parameters: Dict[str, Dict]):
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
    agent(sweep_id)  # type: ignore


sweeps = [
    {
        "project_name": "Embedding-Efficientnet-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/efficientnet_cxl.yml",
        "parameters": {
            "loss_mode": {"values": ["offline/native", "online/soft"]},
            "embedding_size": {"values": [128, 256]},
            "weight_decay": {"values": [0.2, 0.5]},
        },
    },
    {
        "project_name": "Embedding-ConvNeXtV2-CXL-Open",
        "entity": "gorillas",
        "config_path": "./cfgs/convnextv2_cxl.yml",
        "parameters": {
            "loss_mode": {"values": ["offline/native", "online/soft"]},
            "embedding_size": {"values": [128, 256]},
            "weight_decay": {"values": [0.2, 0.5]},
        },
    },
    {
        "project_name": "Embedding-ViT-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/visiontransformer_cxl.yml",
        "parameters": {
            "loss_mode": {"values": ["offline/native", "online/soft"]},
            "embedding_size": {"values": [128, 256]},
            "weight_decay": {"values": [0.2, 0.5]},
        },
    },
]

check_sweep_configs(sweeps)

for current_sweep in sweeps:
    print(f"Running sweep: {current_sweep['project_name']}")
    try:
        run_sweep(**current_sweep)
    except Exception as e:
        print(f"Error running sweep: {current_sweep['project_name']}")
        print(e)
        continue
