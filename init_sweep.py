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
        "program": "train.py",  # Note: not the sweep file, but the training script
        "name": project_name,
        "method": "bayes",  # Specify the search method (random search in this case)
        "metric": {
            "goal": "maximize",
            "name": "cxl_train/val/embeddings/knn5_crossvideo/accuracy",
        },  # Specify the metric to optimize
        "parameters": parameters,
        "command": ["${interpreter}", "${program}", "${args}", "--config_path", config_path],
    }
    sweep_id = sweep(sweep=sweep_config, project=project_name, entity=entity)
    # Print the sweep ID directly
    print(f"SWEEP_PATH={entity}/{project_name}/{sweep_id}")
    agent(sweep_id)


sweeps = [
    {
        "project_name": "Embedding-VitLarge-CXL-OpenSet",
        "entity": "gorillas",
        "config_path": "./cfgs/visiontransformer_cxl.yml",
        "parameters": {
            "dropout_p": {
                "max": 0.5,
                "min": 0.0,
                "distribution": "uniform",
            },
            "l2_alpha": {
                "mu": -3,
                "sigma": 1.0,
                "distribution": "log_normal",
            },
            "l2_beta": {
                "mu": -4.6,
                "sigma": 1.0,
                "distribution": "log_normal",
            },
            "start_lr": {
                "mu": -11.0,
                "sigma": 1.0,
                "distribution": "log_normal",
            },
            "margin": {
                "max": 1.5,
                "min": 0.0,
                "distribution": "uniform",
            },
            "embedding_size": {
                "values": [128, 256, 512],
                "distribution": "categorical",
            },
            # "batch_size": {
            #     "values": [4, 8],
            #     "distribution": "categorical",
            # },
            "loss_mode": {
                "values": [
                    # "softmax/arcface/l2sp",
                    # "softmax/arcface",
                    "online/soft/l2sp",
                    "online/soft",
                    "offline/l2sp",
                    "offline",
                ],
                "distribution": "categorical",
            },
            # "force_nlet_builder": {
            #     "values": ["quadlet", "None"],
            #     "distribution": "categorical",
            # },
            "dataset_class": {
                "values": [
                    "gorillatracker.datasets.cxl.CXLDataset",
                    "gorillatracker.datasets.cxl.CrossEncounterCXLDataset",
                    "gorillatracker.datasets.cxl.HardCrossEncounterCXLDataset",
                ],
                "distribution": "categorical",
            },
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
