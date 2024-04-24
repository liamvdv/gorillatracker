import logging

import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import gorillatracker.ssl_pipeline.contrastive_sampler as contrastive_sampler
import gorillatracker.type_helper as gtypes
from gorillatracker.data_modules import TripletDataModule
from gorillatracker.datasets.cxl import CXLDataset
from gorillatracker.ssl_pipeline.ssl_dataset import SSLDataset, build_triplet

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSLDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        transforms: gtypes.TensorTransform = lambda x: x,
        training_transforms: gtypes.TensorTransform = lambda x: x,
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
            self.setup_val()
        elif stage == "test":
            raise NotImplementedError("test not yet supported by data module.")
        elif stage == "validate":
            self.setup_val()
        elif stage == "predict":
            self.predict = None
            raise NotImplementedError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    # TODO(memben)
    def setup_val(self) -> None:
        self.triplet_data_module = TripletDataModule(
            "/workspaces/gorillatracker/data/splits/derived_data-cxl-yolov8n_gorillabody_ybyh495y-body_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
            dataset_class=CXLDataset,
            batch_size=self.batch_size,
            transforms=transforms.Compose([CXLDataset.get_transforms(), self.transforms]),
            training_transforms=self.training_transforms,
        )
        self.triplet_data_module.setup("fit")

    def train_dataloader(self) -> DataLoader[gtypes.Nlet]:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    # TODO(memben)
    def val_dataloader(self) -> gtypes.BatchNletDataLoader:
        return self.triplet_data_module.val_dataloader()

    def test_dataloader(self) -> gtypes.BatchNletDataLoader:
        raise NotImplementedError

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        raise NotImplementedError

    def collate_fn(self, batch: list[gtypes.Nlet]) -> gtypes.NletBatch:
        ids = tuple(nlet.ids for nlet in batch)
        labels = tuple(nlet.labels for nlet in batch)
        values = tuple(torch.stack(nlet.values) for nlet in batch)
        return ids, values, labels


if __name__ == "__main__":
    dm = SSLDataModule(transforms=transforms.ToTensor())
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
