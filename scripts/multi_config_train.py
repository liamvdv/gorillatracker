import argparse
import json
import os
import pathlib
from typing import Any, Dict

import pandas as pd
import wandb
import yaml

import subprocess
from multiprocessing import Process


def check_experiment_configs(configs: list[dict[str, Any]]) -> None:
    for config in configs:
        assert (
            len(config["project_name"].split("-")) >= 4
        ), "Project name must be of the form <Function>-<Backbone>-<Dataset>-<Set-Type>"
        get_config(config["config_path"])
        print(f"Config file {config['config_path']} is valid")


def get_config(config_path: str) -> Dict[str, Any]:
    assert os.path.isfile(config_path), f"Config file not found at {config_path}"
    assert config_path.endswith(".yml") or config_path.endswith(".yaml"), "Config file must be YAML"
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def get_num_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stdout=subprocess.PIPE, check=True
        )
        gpus = result.stdout.decode("utf-8").strip().split("\n")
        return len(gpus)
    except subprocess.CalledProcessError as e:
        print("Failed to get GPU information:", e)
        return 0


def run_agent_with_gpu(sweep_id, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    wandb.agent(sweep_id, count=1)


def run_experiment(
    project_name: str, config_path: str, parameters: Dict[str, Any], sweep_parameters: Dict[str, Any]
) -> None:

    sweep_name = parameters.pop("sweep_name")
    # Construct the command with parameter overrides
    params_str = " ".join([f"--{key} {value}" for key, value in parameters.items()])

    print(f"Running experiment '{project_name}' with overridden parameters: {parameters}")
    print(f"Sweep parameters: {sweep_parameters}")

    if sweep_parameters:
        params_array = params_str.split(" ")
        sweep_config = {
            "program": "./train.py",  # Note: not the sweep file, but the training script
            "name": sweep_name,
            "method": "bayes",  # Specify the search method (Bayesian optimization in this case)
            "metric": {
                "goal": "maximize",
                "name": "aggregated/cxlkfold/val/embeddings/knn5/accuracy_max",
            },  # Specify the metric to optimize
            "parameters": sweep_parameters,
            "command": ["${interpreter}", "${program}", "${args}", "--config_path", config_path, *params_array],
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name, entity="gorillas")

        sweep_path = f"gorillas/{project_name}/{sweep_id}"
        print("SWEEP_PATH=" + sweep_path)

        num_gpus = get_num_gpus()
        agents_per_gpu = 2  # Number of agents to run per GPU
        processes = []

        if num_gpus == 0:
            print("No GPUs found. Exiting.")
            return

        for gpu_id in range(num_gpus):
            for _ in range(agents_per_gpu):
                p = Process(target=run_agent_with_gpu, args=(sweep_id, gpu_id))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        save_best_run_results(sweep_path, "aggregated/cxlkfold/val/embeddings/knn5/accuracy_max")

    else:
        command = f"python train.py {params_str} --config_path {config_path}"
        os.system(command)
    print(f"Experiment '{project_name}' has been executed with overridden parameters: {parameters}")


def save_best_run_results(sweep_path: str, metric_to_optimize: str) -> None:
    api = wandb.Api()
    sweep_instance = api.sweep(sweep_path)
    best_run = sweep_instance.best_run(order=metric_to_optimize)
    best_run_metrics = best_run.summary
    best_run_config = best_run.config

    best_run_metrics = dict(best_run_metrics)
    best_run_config = dict(best_run_config)
    best_run_config["sweep_path"] = sweep_path
    best_run_config["wandb_link"] = f"https://wandb.ai/{sweep_path}/runs/{best_run.id}"

    metrics_log_path = pathlib.Path(f"results/{sweep_path}/metrics.json")
    config_path = pathlib.Path(f"results/{sweep_path}/config.json")
    metrics_log_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([best_run_metrics]).to_json(metrics_log_path, index=False)
    pd.DataFrame([best_run_config]).to_json(config_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple experiments with different configurations")
    parser.add_argument("--config_path", type=str, help="path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        json_file = json.load(file)
        experiments = json_file["experiments"]
        global_parameters = json_file["global"]
        global_sweep_parameters = json_file.get("global_sweep_parameters", {})

    check_experiment_configs(experiments)

    for current_experiment in experiments:
        print(f"Running experiment: {current_experiment['project_name']}")
        current_experiment["parameters"] = {**current_experiment["parameters"], **global_parameters}
        current_experiment["sweep_parameters"] = {
            **current_experiment.get("sweep_parameters", {}),
            **global_sweep_parameters,
        }
        try:
            run_experiment(**current_experiment)
        except Exception as e:
            print(f"Error running experiment: {current_experiment['project_name']}")
            print(e)
            continue
