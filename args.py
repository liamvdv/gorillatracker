from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

from simple_parsing import field, list_field


@dataclass(kw_only=True)
class TrainingArgs:
    """
    Argument class for use with simple_parsing that handles the basics of most LLM training scripts. Subclass this to add more arguments. TODO: change this
    """

    # Device Arguments
    accelerator: Literal["cuda", "cpu", "tpu", "mps"] = field(default="cuda")
    num_devices: int = field(default=1)
    distributed_strategy: Literal["ddp", "fsdp", "auto", None] = field(default=None)
    force_deterministic: bool = field(default=False)
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = field(default="bf16-mixed")
    compile: bool = field(default=False)
    workers: int = field(default=4)

    # Model and Training Arguments
    run_name: str = field(default="template-test")
    wandb_tags: List[str] = list_field(default=["template"])
    model_name_or_path: str = field(default="EfficientNetV2")
    saved_checkpoint_path: Union[str, None] = field(default=None)
    resume: bool = field(default=False)
    fast_dev_run: bool = field(default=True)
    profiler: Literal["simple", "advanced", "pytorch", None] = field(default=None)
    offline: bool = field(default=True)
    data_preprocessing_only: bool = field(default=False)
    seed: Union[int, None] = field(default=42)
    debug: bool = field(default=False)
    from_scratch: bool = field(default=False)
    early_stopping_patience: int = 3
    embedding_size: int = 256

    learning_rate: float = field(default=0.001)
    weight_decay: float = field(default=0.1)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.999)
    epsilon: float = field(default=1e-8)

    lr_schedule: Literal["linear", "cosine", "exponential", "reduce_on_plateau"] = field(default="linear")
    warmup_epochs: int = field(default=1)
    lr_rate: float = field(default=0.256)
    lr_decay: float = field(default=0.97)
    lr_decay_interval: int = field(default=3)
    margin: float = field(default=0.5)
    loss_mode: Literal["offline", "online/soft", "online/hard", "online/semi-hard"] = field(default="offline")

    batch_size: int = field(default=8)
    grad_clip: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)
    max_epochs: int = field(default=300)
    val_before_training: bool = field(default=False)
    only_val: bool = field(default=False)
    save_interval: float = field(default=10)
    embedding_save_interval: int = field(default=1)

    # Config and Data Arguments
    dataset_class: str = field(default="gorillatracker.datasets.mnist.MNISTDataset")
    data_dir: Path = field(default="./mnist")
    # Add any additional fields as needed.

    def __post_init__(self):
        assert self.num_devices > 0
        assert self.batch_size > 0
        assert self.gradient_accumulation_steps > 0

        if self.grad_clip <= 0:
            self.grad_clip = None
