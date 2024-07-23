from typing import Any, Literal

import torch
import wandb
from lightning import Callback, LightningModule, Trainer


class BestMetricLogger(Callback):
    def __init__(self, metric_name: str, mode: Literal["max", "min"] = "max") -> None:
        super().__init__()
        self.metric_name = metric_name
        self.best_value = float("-inf")
        self.best_metrics: dict[str, Any] = {}
        self.mode = mode

    def check(self, current_value: torch.Tensor) -> bool:
        if self.mode == "max":
            return current_value.item() > self.best_value
        else:
            return current_value.item() < self.best_value

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Get the metrics from the last validation step
        metrics = trainer.callback_metrics
        current_value = metrics.get(self.metric_name)

        if current_value is not None:
            if self.check(current_value):
                self.best_value = current_value.clone().item()
                self.best_metrics = metrics

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.best_metrics:
            wandb.log({f"{self.metric_name}_max": self.best_value})
            for key, value in self.best_metrics.items():
                wandb.log({f"{key}_max": value.item()})
