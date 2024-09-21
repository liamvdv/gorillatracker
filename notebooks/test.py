from gorillatracker.model.wrappers_ssl import MoCoWrapper
from gorillatracker.utils.embedding_generator import generate_embeddings, df_from_predictions
from pathlib import Path
from gorillatracker.data.nlet_dm import NletDataModule
from gorillatracker.data.nlet import build_onelet, SupervisedDataset, SupervisedKFoldDataset
from torchvision.transforms import Resize, Normalize, Compose
import pandas as pd


def get_moco_model(
    checkpoint_path: str = "/workspaces/gorillatracker/models/ssl/moco-accuracy-0.58.ckpt",
) -> MoCoWrapper:
    return MoCoWrapper.load_from_checkpoint(checkpoint_path=checkpoint_path, data_module=None, wandb_run=None)


def get_model_transforms(model):
    resize = getattr(model, "data_resize_transform", (224, 224))
    model_transforms = Resize(resize)
    normalize_transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    use_normalization = getattr(model, "use_normalization", True)
    # NOTE(liamvdv): normalization_mean, normalization_std are always default.
    if use_normalization:
        model_transforms = Compose([model_transforms, normalize_transform])
    return model_transforms


model = get_moco_model()

# TODO(liamvdv): @robert: why filtered? Worauf sind die Dataset Stats?
BRISTOL = Path(
    "/workspaces/gorillatracker/data/supervised/bristol/cross_encounter_validation/cropped_frames_square_filtered"
)
SPAC = Path("/workspaces/gorillatracker/data/supervised/cxl_all/face_images_square")

data_module = NletDataModule(
    data_dir=SPAC,
    dataset_class=SupervisedDataset,
    nlet_builder=build_onelet,
    batch_size=64,
    workers=10,
    model_transforms=get_model_transforms(model),
    training_transforms=lambda x: x,
    dataset_names=["Showcase"],
)

data_module.setup("validate")
dls = data_module.val_dataloader()  # val for transforms
assert len(dls) == 1
dl = dls[0]

preds = generate_embeddings(model, dl)
df = df_from_predictions(preds)
