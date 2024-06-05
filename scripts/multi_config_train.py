import os
from typing import Any, Dict
import yaml
import json


def check_experiment_configs(configs: list[dict[str, Any]]) -> None:
    for config in configs:
        assert (
            len(config["project_name"].split("-")) >= 4
        ), "Project name must be of the form <Function>-<Backbone>-<Dataset>-<Set-Type>"
        get_config(config["config_path"])


def get_config(config_path: str) -> Dict[str, Any]:
    assert os.path.isfile(config_path), f"Config file not found at {config_path}"
    assert config_path.endswith(".yml") or config_path.endswith(".yaml"), "Config file must be YAML"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def run_experiment(project_name: str, config_path: str, parameters: Dict[str, Any]) -> None:
    # Construct the command with parameter overrides
    params_str = " ".join([f"--{key} {value}" for key, value in parameters.items()])
    command = f"python train.py {params_str} --config_path {config_path}"
    os.system(command)
    print(f"Experiment '{project_name}' has been executed with overridden parameters: {parameters}")


if __name__ == "__main__":
    with open("scripts/multi_configs/kn.json", "r") as file:
        json_file = json.load(file)
        experiments = json_file["experiments"]
        global_parameters = json_file["global"]

    check_experiment_configs(experiments)

    for current_experiment in experiments:
        print(f"Running experiment: {current_experiment['project_name']}")
        current_experiment["parameters"] = {**current_experiment["parameters"], **global_parameters}
        try:
            run_experiment(**current_experiment)
        except Exception as e:
            print(f"Error running experiment: {current_experiment['project_name']}")
            print(e)
            continue
