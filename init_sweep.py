import os
from typing import Dict, Union, cast

import yaml
from wandb import agent, sweep

# Set your default config
config_path = "./cfgs/swinv2_large_cxl.yml"
# Assuming `config_path` is the path to your YAML config file
assert os.path.isfile(config_path), f"Config file not found at {config_path}"
assert config_path.endswith(".yml") or config_path.endswith(".yaml"), "Config file must be YAML"
config_dict: Dict[str, Union[str, int, float, bool]] = dict()
with open(config_path, "r") as file:
    config_dict = yaml.safe_load(file)

project_name = cast(str, config_dict["project_name"])

assert len(project_name.split("-")) >= 4, "Project name must be of the form <Function>-<Backbone>-<Dataset>-<Set-Type>"

# Define your sweep configuration
sweep_config = {
    "program": "./train.py",  # Note: not the sweep file, but the training script
    "name": project_name,
    "method": "grid",  # Specify the search method (random search in this case)
    "metric": {"goal": "minimize", "name": "val/loss_epoch"},  # Specify the metric to optimize
    "parameters": {
        # "param1": {"min": 1, "max": 10},  # Define parameter search space
        "weight_decay": {"values": [1e-2, 1e-3, 1e-4, 1e-5]},
        "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
        # Add other parameters as needed
    },
    "command": ["${interpreter}", "${program}", "${args}", "--config_path", config_path],
}
# Initialize the sweep
project_name = "Embedding-SwinV2Large-CXL-Open"
entity = "gorillas"
sweep_id = sweep(sweep=sweep_config, project=project_name, entity=entity)
# Print the sweep ID directly
print(f"SWEEP_PATH={entity}/{project_name}/{sweep_id}")
agent(sweep_id)  # type: ignore
