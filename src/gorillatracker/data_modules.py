import logging
from typing import Any, Callable, Literal, Optional, Type

import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import gorillatracker.type_helper as gtypes
from gorillatracker.data_loaders import QuadletDataLoader, SimpleDataLoader, TripletDataLoader

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NletDataModule(L.LightningDataModule):
    """
    Base class for triplet/quadlet data modules, implementing shared functionality.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        dataset_class: Optional[Type[Dataset[Any]]] = None,
        transforms: Optional[gtypes.Transform] = None,
        training_transforms: Optional[gtypes.Transform] = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.training_transforms = training_transforms
        self.dataset_class = dataset_class
        self.data_dir = data_dir
        self.batch_size = batch_size

    def get_dataloader(self) -> Any:
        raise Exception("logic error, ask liamvdv")

    def setup(self, stage: str) -> None:
        assert self.dataset_class is not None, "dataset_class must be set before calling setup"
        logger.info(
            f"setup {stage} for Dataset {self.dataset_class.__name__} via Dataload {self.get_dataloader().__name__}"
        )

        if stage == "fit":
            self.train = self.dataset_class(self.data_dir, partition="train", transform=transforms.Compose([self.transforms, self.training_transforms]))  # type: ignore
            self.val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)  # type: ignore
        elif stage == "test":
            self.test = self.dataset_class(self.data_dir, partition="test", transform=self.transforms)  # type: ignore
        elif stage == "validate":
            self.val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)  # type: ignore
        elif stage == "predict":
            # TODO(liamvdv): delay until we know how things should look.
            # self.predict = None
            raise ValueError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self) -> gtypes.BatchNletDataLoader:
        return self.get_dataloader()(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> gtypes.BatchNletDataLoader:
        return self.get_dataloader()(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> gtypes.BatchNletDataLoader:
        return self.get_dataloader()(self.test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        # return self.get_dataloader()(self.predict, batch_size=self.batch_size, shuffle=False)
        raise NotImplementedError("predict_dataloader not implemented")

    def teardown(self, stage: str) -> None:
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass

    def get_num_classes(self, mode: Literal["train", "val", "test"]) -> int:  # HACK
        if mode == "train":
            train = self.dataset_class(self.data_dir, partition="train", transform=transforms.Compose([self.transforms, self.training_transforms]))  # type: ignore
            return train.get_num_classes()  # type: ignore
        elif mode == "val":
            val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)  # type: ignore
            return val.get_num_classes()  # type: ignore
        elif mode == "test":
            test = self.dataset_class(self.data_dir, partition="test", transform=self.transforms)  # type: ignore
            return test.get_num_classes()  # type: ignore
        else:
            raise ValueError(f"unknown mode '{mode}'")


class TripletDataModule(NletDataModule):
    def get_dataloader(self) -> Callable[[Dataset[Any], int, bool], gtypes.BatchTripletDataLoader]:
        return TripletDataLoader


class QuadletDataModule(NletDataModule):
    def get_dataloader(self) -> Callable[[Dataset[Any], int, bool], gtypes.BatchQuadletDataLoader]:
        return QuadletDataLoader


class SimpleDataModule(NletDataModule):
    def get_dataloader(self) -> Callable[[Dataset[Any], int, bool], gtypes.BatchSimpleDataLoader]:
        return SimpleDataLoader
