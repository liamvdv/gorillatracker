import concurrent.futures
import csv
import random
import subprocess
import fcntl
import time
from typing import Optional

import timm
import torch
import wandb

# NOTE(rob2u): a simple context manager to lock files
# This avoids the edge case that a task is executed twice
class FileLock:
    def __init__(self, file):
        self.file = file 

    def __enter__(self):
        fcntl.flock(self.file, fcntl.LOCK_EX)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def read_backbones_todo_csv(file_path: str) -> list[str]:
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        backbones = [row[0] for row in reader][1:]  # Skip header
    return backbones


def get_existing_runs() -> list[str]:
    api = wandb.Api()

    project_path = "gorillas/EVAL3-ALL-CXL-OpenSet"
    runs = api.runs(project_path)
    run_names = [run.name.split("-")[-1] for run in runs]  # 000-eval-<backbone_name> -> <backbone_name>
    return run_names


def get_command(backbone_name: str) -> Optional[list[str]]:
    try:
        model = timm.create_model(backbone_name, pretrained=True)
        model = model.eval()
    except RuntimeError as e:
        print(f"Error loading {backbone_name}: {e}")
        return None

    data_config = timm.data.resolve_model_data_config(model)
    is_normalize_matching = data_config["mean"] == (0.485, 0.456, 0.406) and data_config["std"] == (0.229, 0.224, 0.225)
    if not is_normalize_matching:
        print(f"Warning: data normalization does not match for {backbone_name}")
        print(f"Got: {data_config['mean']} {data_config['std']}")

    features_ = model.forward_features(torch.randn(1, 3, data_config["input_size"][1], data_config["input_size"][2]))
    embedding_size = model.forward_head(features_, pre_logits=True).shape[1]
    input_size = data_config["input_size"][-1]
    input_size = min(input_size, 768)  # Limit input size to 768
    if input_size != data_config["input_size"][-1]:
        print(f"Warning: input size limited to 768 for {backbone_name}")

    return [
        "python",
        "train.py",
        "--config_path=cfgs/backbone_eval.yml",
        "--only_val=True",
        f"--run_name=000-eval-{backbone_name}",
        f"--embedding_size={embedding_size}",
        f"--data_resize_transform={input_size}",
        f"--model_name_or_path=timm_eval/{backbone_name}",
        f"--normalization_mean={list(data_config['mean'])}",
        f"--normalization_std={list(data_config['std'])}",
    ]


def run_command(backbone_name: str) -> tuple[Optional[str], Optional[str]]:
    with FileLock(open(".lock", "w")):
        print(f"Running {backbone_name}")
        start_time = time.time()
        if backbone_name in get_existing_runs():
            return None, None

        command = get_command(backbone_name)
        if command is None:
            return None, None
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 15: # NOTE(rob2u): give the process at least 15 seconds to start in order to prevent duplicate wandb runs
            time.sleep(15 - int(elapsed_time) + 1)    
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"waiting for {command}")
    stdout, stderr = process.communicate()  # Wait for the process to finish (blocking)
    print(f"finished {command}")
    return stdout.decode(), stderr.decode()


if __name__ == "__main__":
    with open(".lock", "w") as f:
        pass
    max_processes = 6
    backbone_list_all = read_backbones_todo_csv("backbone_names_all.csv")
    backbone_list_done = get_existing_runs()
    backbone_list_todo = list(set(backbone_list_all) - set(backbone_list_done))
    print(f"Backbones to evaluate: {len(backbone_list_todo)}")
    # NOTE(rob2u): we permute the list in order to distribute the load more evenly
    random.shuffle(backbone_list_todo)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(run_command, backbone_name): backbone_name for backbone_name in backbone_list_todo}

        for future in concurrent.futures.as_completed(futures):
            cmd = futures[future]
            try:
                stdout, stderr = future.result()
                if stdout is not None:
                    results.append((cmd, stdout, stderr))
                    print(f"{cmd} completed")
                    # print(f"{cmd} completed with output: {stdout}")
            except Exception as exc:
                print(f"{cmd} generated an exception: {exc}")
