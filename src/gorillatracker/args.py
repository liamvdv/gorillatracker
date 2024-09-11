from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

from simple_parsing import field, list_field


@dataclass(kw_only=True)
class TrainingArgs:
    """
    Argument class for use with simple_parsing that handles the basics of most LLM training scripts. Subclass this to add more arguments. TODO: change this
    """

    # Device Arguments
    accelerator: Literal["cuda", "cpu", "tpu", "mps"] = field(default="cuda")
    num_devices: int = field(default=1)
    distributed_strategy: Literal["ddp", "fsdp", "auto", None] = field(default="auto")
    force_deterministic: bool = field(default=True)
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
    model_name_or_path: str = field(default="timm/efficientnetv2_rw_m")  # TODO
    freeze_backbone: bool = field(default=False)

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
    embedding_id: Literal[
        "linear", "mlp", "linear_norm_dropout", "mlp_norm_dropout", "linear_single_norm_dropout", "none"
    ] = field(default="linear")
    pool_mode: Literal["gem", "gap", "gem_c", "none"] = field(default="none")
    fix_img_size: Optional[int] = field(default=None)

    aug_num_ops: int = field(default=2)
    aug_magnitude: int = field(default=5)

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
    lr_interval: float = field(default=1)  # Fraction of an epoch after which learning rate is updated

    save_model_to_wandb: Union[Literal["all"], bool] = field(default=False)
    stop_saving_metric_name: str = "cxl/val/embeddings/knn5_crossvideo/accuracy"
    stop_saving_metric_mode: Literal["min", "max"] = "max"

    # NTXent Arguments
    temperature: float = field(default=0.5)
    memory_bank_size: int = field(default=0)

    # ArcFace Arguments
    k_subcenters: int = field(default=1)
    s: float = field(default=64.0)

    margin: float = field(default=0.5)
    additive_margin: float = field(default=0.0)
    margin_std: float = field(default=0.05)

    loss_mode: Literal[
        "offline",
        "offline/native",
        "online/soft",
        "online/hard",
        "online/semi-hard",
        "softmax/arcface",
        "softmax/adaface",
        "softmax/elasticface",
        "offline/native/l2sp",
        "offline/l2sp",
        "online/soft/l2sp",
        "online/hard/l2sp",
        "online/semi-hard/l2sp",
        "softmax/arcface/l2sp",
        "softmax/adaface/l2sp",
        "softmax/elasticface/l2sp",
        "distillation/offline/response-based",
        "ntxent",
        "mae_mse",
        "mae_mse/l2sp",
        "mae_mse/arcface",
        "mae_mse/arcface/l2sp",
        "ntxent/l2sp",
    ] = field(default="offline")
    cross_video_masking: bool = field(default=False)
    loss_dist_term: Literal["cosine", "euclidean"] = field(default="euclidean")
    teacher_model_wandb_link: str = field(default="")
    kfold: bool = field(default=False)
    use_focal_loss: bool = field(default=False)
    label_smoothing: float = field(default=0.0)
    use_class_weights: bool = field(default=False)
    use_dist_term: bool = field(default=False)
    use_normalization: bool = field(default=True)
    normalization_mean: str = field(default="[0.485, 0.456, 0.406]")
    normalization_std: str = field(default="[0.229, 0.224, 0.225]")
    use_inbatch_mixup: bool = field(default=False)
    force_nlet_builder: Literal["onelet", "pair", "triplet", "quadlet", "None"] = field(default="None")

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
    additional_val_dataset_classes: Union[list[str], str] = field(default_factory=list)
    additional_val_data_dirs: Union[list[str], str] = field(default_factory=list)
    dataset_names: Union[list[str], str] = field(default_factory=list)
    data_resize_transform: Union[int, None] = field(default=None)

    # SSL Config
    use_ssl: bool = field(default=False)
    tff_selection: Literal["random", "equidistant", "embeddingdistant", "movement"] = field(default="equidistant")
    split_path: Path = field(default=Path("ERROR_PATH_NOT_SET_SEE_ARGS"))
    negative_mining: Literal["random", "overlapping", "social_groups"] = field(default="random")
    n_samples: int = field(default=15)
    feature_types: list[str] = field(default_factory=lambda: ["body"])
    min_confidence: float = field(default=0.5)
    min_images_per_tracking: int = field(default=3)
    width_range: tuple[Union[int, None], Union[int, None]] = field(default=(None, None))
    height_range: tuple[Union[int, None], Union[int, None]] = field(default=(None, None))
    movement_delta: Union[float, None] = field(default=None)
    forced_train_image_count: Union[int, None] = field(default=None)
    multi_gpu_training: bool = field(default=False)

    def __post_init__(self) -> None:
        assert self.num_devices > 0
        assert self.batch_size > 0
        assert self.gradient_accumulation_steps > 0
        assert isinstance(self.grad_clip, float), "automatically set to None if < 0"
        if not (
            self.width_range[0] is None or self.width_range[1] is None or self.width_range[0] <= self.width_range[1]
        ):
            raise ValueError("min_width should be <= max_width")
        if not (
            self.height_range[0] is None or self.height_range[1] is None or self.height_range[0] <= self.height_range[1]
        ):
            raise ValueError("min_height should be <= max_height")
        assert all(
            s in ["body", "face_90", "face_45", "body_with_face"] for s in self.feature_types
        ), "Invalid feature type"
        assert self.tff_selection != "movement" or self.movement_delta is not None, "Combination not allowed"
        if self.grad_clip <= 0:
            self.grad_clip = None
        assert self.lr_interval <= 1, "lr_interval should be <= 1"

        if isinstance(self.additional_val_data_dirs, str):
            self.additional_val_data_dirs = eval(self.additional_val_data_dirs)
        if isinstance(self.additional_val_dataset_classes, str):
            self.additional_val_dataset_classes = eval(self.additional_val_dataset_classes)
        if isinstance(self.dataset_names, str):
            self.dataset_names = eval(self.dataset_names)
