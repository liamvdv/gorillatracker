import numpy as np
from pathlib import Path
from gorillatracker.utils import embedding_generator
from gorillatracker.data.nlet import NletDataModule, build_onelet
from gorillatracker.data.ssl import SSLDataset
from gorillatracker.utils import wandb_loader
from torchvision.transforms import Compose, Normalize, Resize

from gorillatracker.ssl_pipeline.ssl_config import SSLConfig

DATA_DIR = Path("/workspaces/gorillatracker/video_data/cropped-images/2024-04-18")


def generate_perceptional_embeddings(
    wandb_run: str = "https://wandb.ai/gorillas/Embedding-ViTFrozen-CXL-OpenSet/runs/5fuj7iqs",
    split_path: Path = Path(
        "/workspaces/gorillatracker/data/splits/SSL/SSL-10k-100-1000_2024-04-18_percentage-90-5-5_split_20240619_0955.pkl"
    ),
    save_path: Path = Path("/workspaces/gorillatracker/"),
):
    DATASET_CLS = SSLDataset

    # Sample everything
    CONFIG = SSLConfig(
        tff_selection="equidistant",
        negative_mining="random",
        n_samples=100_000,
        feature_types=["body", "face_45", "face_90"],
        min_confidence=0.0,
        min_images_per_tracking=0,
        split_path=split_path,
        width_range=(None, None),
        height_range=(None, None),
    )

    model = wandb_loader.get_model_for_run_url(wandb_run)
    resize = getattr(model, "data_resize_transform", (224, 224))
    model_transforms = Resize(resize)
    normalize_transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    use_normalization = getattr(model, "use_normalization", True)
    if use_normalization:
        model_transforms = Compose([model_transforms, normalize_transform])

    data_module = NletDataModule(
        data_dir=DATA_DIR,
        dataset_class=DATASET_CLS,
        nlet_builder=build_onelet,
        batch_size=64,
        workers=10,
        model_transforms=model_transforms,
        training_transforms=lambda x: x,
        dataset_names=["Inference"],
        ssl_config=CONFIG,
    )

    data_module.setup("fit")

    ids_train, embeddings_train, _ = embedding_generator.generate_embeddings(model, data_module.train_dataloader())
    ids_val, embeddings_val, _ = embedding_generator.generate_embeddings(model, data_module.val_dataloader()[0])
    ids = ids_train + ids_val
    ids = np.array([int(Path(ids).stem) for ids in ids])
    embeddings = np.concatenate([embeddings_train, embeddings_val])
    np.save(save_path / "vit_ids.npy", ids)
    np.save(save_path / "vit_embeddings.npy", embeddings)


if __name__ == "__main__":
    generate_perceptional_embeddings()
