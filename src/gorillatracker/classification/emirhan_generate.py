from pathlib import Path

import numpy as np
from torchvision.transforms import Compose, Normalize, Resize

from gorillatracker.data.nlet import build_onelet
from gorillatracker.data.nlet_dm import NletDataModule
from gorillatracker.data.ssl import SSLDataset
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig
from gorillatracker.utils import embedding_generator
from gorillatracker.model.wrappers_ssl import MoCoWrapper
from gorillatracker.model.wrappers_supervised import BaseModuleSupervised

DATA_DIR = Path("/workspaces/gorillatracker/video_data/cropped-images/2024-04-18")


def get_finetuned_vit() -> MoCoWrapper:
    # ViT Large + DinoV2; finetuned with SSL and MoCo Loss
    # https://wandb.ai/gorillas/Embedding-VitLarge-MoCo-Face-Sweep/runs/rlemhfix
    finetuned = "/workspaces/gorillatracker/models/ssl/vit-large-moco-face-sweep/rlemhfix/checkpoints/epoch-3-cxl-all/val/embeddings/knn_crossvideo_cos/accuracy-0.58.ckpt"
    return MoCoWrapper.load_from_checkpoint(
        checkpoint_path=finetuned,
        data_module=None,
        wandb_run=None,
    )


def get_model() -> BaseModuleSupervised:
    finetuned = "/workspaces/gorillatracker/models/roberts_models/gorillas_models/vit_large_dinov2_bayes/fold-0-epoch-19-cxlkfold/fold-0/val/embeddings/knn5_crossvideo/accuracy-0.63.ckpt"
    return BaseModuleSupervised.load_from_checkpoint(
        checkpoint_path=finetuned,
        data_module=None,
        wandb_run=None,
    )


def generate_perceptional_embeddings(
    split_path: Path = Path(
        "/workspaces/gorillatracker/data/splits/SSL/SSL-all_2024-04-18_percentage-100-0-0_split_20240919_0749.pkl"
        # NOTE(liamvdv): Use for testing
        # "/workspaces/gorillatracker/data/splits/SSL/SSL-1k-woCXL_1k-100-1k_split_20240716_1037.pkl"
    ),
    save_path: Path = Path("/workspaces/gorillatracker/"),
) -> None:
    DATASET_CLS = SSLDataset

    # Sample everything
    CONFIG = SSLConfig(
        tff_selection="equidistant",
        negative_mining="random",
        n_samples=100,
        feature_types=["face_45"],
        min_confidence=0.0,
        min_images_per_tracking=0,
        split_path=split_path,
        width_range=(90, None),
        height_range=(90, None),
    )

    model = get_model()
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
        workers=40,
        model_transforms=model_transforms,
        training_transforms=lambda x: x,
        dataset_names=["Inference"],
        ssl_config=CONFIG,
    )

    data_module.setup("fit")

    ids_train, embeddings_train, labels_train = embedding_generator.generate_embeddings(
        model, data_module.train_dataloader()
    )
    # ids_val, embeddings_val, labels_val = embedding_generator.generate_embeddings(
    #     model, data_module.val_dataloader()[0]
    # )
    # image_ids = ids_train + ids_val
    # ids = np.array([int(Path(ids).stem) for ids in image_ids])
    # labels = np.concatenate([labels_train, labels_val])
    # embeddings = np.concatenate([embeddings_train, embeddings_val])
    np.save(save_path / "vit_ids.npy", ids_train)
    np.save(save_path / "vit_embeddings.npy", embeddings_train)
    np.save(save_path / "vit_labels.npy", labels_train)


if __name__ == "__main__":
    generate_perceptional_embeddings()
