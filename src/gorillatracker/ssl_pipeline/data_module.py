import logging
from typing import List, Optional

import lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

import gorillatracker.type_helper as gtypes
from gorillatracker.data_modules import TripletDataModule
from gorillatracker.datasets.cxl import CXLDataset
from gorillatracker.ssl_pipeline.ssl_dataset import SSLDataset, build_triplet
from gorillatracker.train_utils import _assert_tensor, get_dataset_class

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSLDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        transforms: gtypes.Transform = lambda x: x,
        training_transforms: gtypes.Transform = lambda x: x,
        data_dir: str = "/workspaces/gorillatracker/cropped_images/2024-04-09",
        additional_dataset_class_ids: Optional[List[str]] = None,
        additional_data_dirs: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.training_transforms = training_transforms
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.additional_dataset_class_ids = additional_dataset_class_ids
        self.additional_data_dirs = additional_data_dirs

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = SSLDataset(
                self.data_dir,
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

    # TODO(memben): we want to use SSL Data for validation
    def setup_val(self) -> None:
        print("Using Body-Image Validation Set")
        dataset_classes = None
        transforms_list = None
        if self.additional_dataset_class_ids is not None:
            assert self.additional_data_dirs is not None, "additional_data_dirs must be set"
            dataset_classes = [get_dataset_class(cls_id) for cls_id in self.additional_dataset_class_ids]
            transforms_list = []
            for cls in dataset_classes:
                transforms_list.append(
                    Compose(
                        [
                            cls.get_transforms() if hasattr(cls, "get_transforms") else ToTensor(),
                            _assert_tensor,
                            self.transforms,
                        ]
                    )
                )

        dataset_class = CXLDataset
        self.val_data_module = TripletDataModule(
            "/workspaces/gorillatracker/data/splits/derived_data-cxl-yolov8n_gorillabody_ybyh495y-body_images-openset-reid-val-0-test-0-mintraincount-3-seed-42-train-50-val-25-test-25",
            dataset_class=dataset_class,
            batch_size=self.batch_size,
            transforms=transforms.Compose([dataset_class.get_transforms(), self.transforms]),
            training_transforms=self.training_transforms,
            additional_dataset_classes=dataset_classes,
            additional_data_dirs=self.additional_data_dirs,
            additional_transforms=transforms_list,
        )
        self.val_data_module.setup("fit")

    def train_dataloader(self) -> DataLoader[gtypes.Nlet]:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    # TODO(memben): we want to use SSL Data for validation
    def val_dataloader(self) -> List[gtypes.BatchNletDataLoader]:
        return self.val_data_module.val_dataloader()

    def test_dataloader(self) -> gtypes.BatchNletDataLoader:
        raise NotImplementedError

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        raise NotImplementedError

    def collate_fn(self, batch: list[gtypes.Nlet]) -> gtypes.NletBatch:
        ids = tuple(nlet[0] for nlet in batch)
        values = tuple(torch.stack(nlet[1]) for nlet in batch)
        labels = tuple(nlet[2] for nlet in batch)
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
