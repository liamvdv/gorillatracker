from wandb import agent, sweep

# Set your default config
config_path = "./cfgs/resnet18_cxl.yml"
# Define your sweep configuration
sweep_config = {
    "program": "./train.py",  # Note: not the sweep file, but the training script
    "name": "test",
    "method": "grid",  # Specify the search method (random search in this case)
    "metric": {"goal": "minimize", "name": "val/loss"},  # Specify the metric to optimize
    "parameters": {
        # "param1": {"min": 1, "max": 10},  # Define parameter search space
        "loss_mode": {"values": ["offline", "offline/native", "online/soft", "online/semi-hard"]},
        # Add other parameters as needed
    },
    "command": ["${interpreter}", "${program}", "${args}", "--config_path", config_path],
}
# Initialize the sweep
project_name = "test_losses"
entity = "gorillas"
sweep_id = sweep(sweep=sweep_config, project=project_name, entity=entity)
# Print the sweep ID directly
print(f"SWEEP_PATH={entity}/{project_name}/{sweep_id}")
agent(sweep_id)
