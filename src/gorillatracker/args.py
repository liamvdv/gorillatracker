from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Union

from simple_parsing import field, list_field


@dataclass(kw_only=True)  # type: ignore
class TrainingArgs:
    """
    Argument class for use with simple_parsing that handles the basics of most LLM training scripts. Subclass this to add more arguments. TODO: change this
    """

    # Device Arguments
    accelerator: Literal["cuda", "cpu", "tpu", "mps"] = field(default="cuda")
    num_devices: int = field(default=1)
    distributed_strategy: Literal["ddp", "fsdp", "auto", None] = field(default=None)
    force_deterministic: bool = field(default=False)
    precision: Literal[
        "32-true",
        "16-mixed",
        "bf16-mixed",
        "16-true",
        "transformer-engine-float16",
        "transformer-engine",
        "int8-training",
        "int8",
        "fp4",
        "nf4",
        "",
    ] = field(default="bf16-mixed")
    compile: bool = field(default=False)
    workers: int = field(default=4)

    # Model and Training Arguments
    project_name: str = field(default="")
    run_name: str = field(default="")
    wandb_tags: List[str] = list_field(default=["template"])
    model_name_or_path: str = field(default="EfficientNetV2")
    use_wildme_model: bool = field(default=False)
    saved_checkpoint_path: Union[str, None] = field(default=None)
    resume: bool = field(default=False)
    fast_dev_run: bool = field(default=True)
    profiler: Union[Literal["simple", "advanced", "pytorch"], None] = field(default=None)
    offline: bool = field(default=True)
    data_preprocessing_only: bool = field(default=False)
    seed: Union[int, None] = field(default=42)
    debug: bool = field(default=False)
    from_scratch: bool = field(default=False)
    early_stopping_patience: int = 3
    min_delta: float = field(default=0.01)
    embedding_size: int = 256
    dropout_p: float = field(default=0.0)
    use_quantization_aware_training: bool = field(default=False)

    # Optimizer Arguments
    weight_decay: float = field(default=0.1)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.999)
    epsilon: float = field(default=1e-8)

    # L2SP Arguments
    l2_alpha: float = field(default=0.1)
    l2_beta: float = field(default=0.01)
    path_to_pretrained_weights: Union[str, None] = field(default=None)

    lr_schedule: Literal["linear", "cosine", "exponential", "reduce_on_plateau", "constant"] = field(default="constant")
    warmup_mode: Literal["linear", "cosine", "exponential", "constant"] = field(default="constant")
    warmup_epochs: int = field(default=0)
    initial_lr: float = field(default=1e-5)
    start_lr: float = field(default=1e-5)
    end_lr: float = field(default=1e-5)
    stepwise_schedule: bool = field(default=False)

    save_model_to_wandb: bool = field(default=False)

    margin: float = field(default=0.5)
    s: float = field(default=64.0)
    delta_t: int = field(default=100)
    mem_bank_start_epoch: int = field(default=2)
    lambda_membank: float = field(default=0.5)
    loss_mode: Literal[
        "offline",
        "offline/native",
        "online/soft",
        "online/hard",
        "online/semi-hard",
        "softmax/arcface",
        "softmax/vpl",
        "offline/native/l2sp",
        "online/soft/l2sp",
        "online/hard/l2sp",
        "online/semi-hard/l2sp",
        "softmax/arcface/l2sp",
        "softmax/vpl/l2sp",
    ] = field(default="offline")
    kfold: bool = field(default=False)

    batch_size: int = field(default=8)
    grad_clip: Union[float, None] = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)
    max_epochs: int = field(default=300)
    val_check_interval: float = field(default=1.0)
    check_val_every_n_epoch: int = field(default=1)
    val_before_training: bool = field(default=False)
    only_val: bool = field(default=False)
    save_interval: float = field(default=10)
    embedding_save_interval: int = field(default=1)
    knn_with_train: bool = field(default=True)
    plugins: List[str] = list_field(default=None)

    # Config and Data Arguments
    dataset_class: str = field(default="gorillatracker.datasets.mnist.MNISTDataset")
    data_dir: Path = field(default=Path("./mnist"))
    additional_val_dataset_classes: Union[List[str], None] = field(default=None)
    additional_val_data_dirs: Union[List[str], None] = field(default=None)
    data_resize_transform: Union[int, None] = field(default=None)

    # SSL Config
    use_ssl: bool = field(default=False)
    tff_selection: Literal["random", "equidistant"] = field(default="equidistant")
    n_videos: int = field(default=200)
    n_samples: int = field(default=15)
    feature_types: list[str] = field(default_factory=lambda: ["body"])
    min_confidence: float = field(default=0.5)
    min_images_per_tracking: int = field(default=3)

    def __post_init__(self) -> None:
        assert self.num_devices > 0
        assert self.batch_size > 0
        assert self.gradient_accumulation_steps > 0
        assert isinstance(self.grad_clip, float), "automatically set to None if < 0"
        if self.grad_clip <= 0:
            self.grad_clip = None
