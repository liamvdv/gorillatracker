import os
from typing import Any, Dict, Union
import argparse
import json

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


def run_sweep(project_name: str, entity: str, config_path: str, parameters: Dict[str, Dict[str, Any]], sweep_name: str = "", method: str = "grid", sweep_metric: str = "aggregated/cxlkfold/val/embeddings/knn5_filter/accuracy_max") -> None:
    if sweep_name == "":
        sweep_name = project_name
    sweep_config = {
        "program": "./train.py",  # Note: not the sweep file, but the training script
        "name": sweep_name,
        "method": method,  # Specify the search method (random search in this case)
        "metric": {
            "goal": "maximize",
            "name": sweep_metric,
        },  # Specify the metric to optimize
        "parameters": parameters,
        "command": ["${interpreter}", "${program}", "${args}", "--config_path", config_path],
    }
    sweep_id = sweep(sweep=sweep_config, project=project_name, entity=entity)
    # Print the sweep ID directly
    print(f"SWEEP_PATH={entity}/{project_name}/{sweep_id}")
    agent(sweep_id)



if __name__ == "__main__":
    sweeps = [
        {
            "project_name": "Embedding-EfficientNetRWM-CXL-OpenSet",
            "entity": "gorillas",
            "config_path": "./cfgs/efficientnet_rw_m_cxl.yml",
            "parameters": {
                "weight_decay": {
                    "values": [
                        0.7,
                        0.5,
                        0.1,
                    ]
                },
                "dropout_p": {"values": [0.5, 0.3, 0.1]},
                "start_lr": {"values": [1e-3, 1e-4, 1e-5]},
                "end_lr": {"values": [1e-5, 1e-6, 1e-7]},
            },
        },
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_config_file", type=str, default=None, help="Path to sweep configs directory")
    args = parser.parse_args()
    if args.sweep_config_file:
        sweeps = json.load(open(args.sweep_config_file, "r"))

    check_sweep_configs(sweeps)

    for current_sweep in sweeps:
        print(f"Running sweep: {current_sweep['project_name']}")
        try:
            run_sweep(**current_sweep)  # type: ignore
        except Exception as e:
            print(f"Error running sweep: {current_sweep['project_name']}")
            print(e)
            continue