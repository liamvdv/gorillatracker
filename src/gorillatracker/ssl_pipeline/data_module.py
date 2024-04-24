import logging

import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from gorillatracker.datasets.cxl import CXLDataset
import gorillatracker.ssl_pipeline.contrastive_sampler as contrastive_sampler
import gorillatracker.type_helper as gtypes
from gorillatracker.data_modules import TripletDataModule
from gorillatracker.ssl_pipeline.dataset import SSLDataset, build_triplet
from gorillatracker.transform_utils import SquarePad

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSLDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        transforms: gtypes.Transform = lambda x: x,
        training_transforms: gtypes.Transform = lambda x: x,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.training_transforms = training_transforms
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        print("Setting up data module ", stage)
        if stage == "fit":
            train_sampler = contrastive_sampler.WIP_clique_sampler()
            self.train = SSLDataset(
                train_sampler,
                build_triplet,
                "train",
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )
        elif stage == "test":
            raise NotImplementedError("test not yet supported by data module.")
        elif stage == "validate":
            raise NotImplementedError("validate not yet supported by data module.")
        elif stage == "predict":
            self.predict = None
            raise NotImplementedError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("fit")
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn) # type: ignore

    # TODO(memben)
    def val_dataloader(self) -> gtypes.BatchNletDataLoader:
        data_dir = "/workspaces/gorillatracker/data/splits/ground_truth-cxl-face_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25"
        dm = TripletDataModule(
            data_dir,
            dataset_class=CXLDataset,
            batch_size=self.batch_size,
            transforms=self.transforms,
            training_transforms=self.training_transforms,
        )
        dm.setup("fit")
        return dm.train_dataloader()

    def test_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("test")
        raise NotImplementedError("test_dataloader not implemented")

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("predict")
        raise NotImplementedError("predict_dataloader not implemented")

    def collate_fn(self, batch: list[gtypes.Nlet]) -> gtypes.NletBatch:
        ids = tuple(nlet.ids for nlet in batch)
        labels = tuple(nlet.labels for nlet in batch)
        values = tuple(torch.stack(nlet.values) for nlet in batch)
        return ids, values, labels

    @classmethod
    def get_transforms(cls) -> gtypes.Transform:
        return transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(224),
                transforms.ToTensor(),
                # Uniform input, you may choose higher/lower sizes.
            ]
        )


if __name__ == "__main__":
    dm = SSLDataModule(transforms=SSLDataModule.get_transforms())
    print("Data Module created")
    dm.setup("fit")
    print("Data Module setup")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print("Data Loaders created")
    for batch in train_loader:
        print(batch)
        break
    for batch in val_loader:
        print(batch)
        break
