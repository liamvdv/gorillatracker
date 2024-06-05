import logging
from typing import Any, Callable, List, Literal, Optional, Type

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
        additional_dataset_classes: Optional[List[Type[Dataset[Any]]]] = None,
        additional_data_dirs: Optional[List[str]] = None,
        additional_transforms: Optional[List[gtypes.Transform]] = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.training_transforms = training_transforms
        self.dataset_class = dataset_class
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.additional_dataset_classes = additional_dataset_classes
        self.additional_data_dirs = additional_data_dirs
        self.additional_transforms = additional_transforms

        assert (additional_dataset_classes is None and additional_data_dirs is None) or len(
            additional_dataset_classes  # type: ignore
        ) == len(
            additional_data_dirs  # type: ignore
        ), "additional_dataset_classes and additional_data_dirs must have the same length"

    def get_dataloader(self) -> Any:
        raise Exception("logic error, ask liamvdv")

    def setup(self, stage: str) -> None:
        assert self.dataset_class is not None, "dataset_class must be set before calling setup"
        logger.info(
            f"setup {stage} for Dataset {self.dataset_class.__name__} via Dataload {self.get_dataloader().__name__}"
        )

        if stage == "fit":
            self.train = self.dataset_class(
                self.data_dir,
                partition="train",
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )  # type: ignore
            self.val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)  # type: ignore
            if self.additional_dataset_classes is not None:
                self.val_list = [self.val]
                for data_dir, dataset_class, transform in zip(
                    self.additional_data_dirs, self.additional_dataset_classes, self.additional_transforms  # type: ignore
                ):
                    self.val_list.append(dataset_class(data_dir, partition="val", transform=transform))  # type: ignore
        elif stage == "test":
            self.test = self.dataset_class(self.data_dir, partition="test", transform=self.transforms)  # type: ignore
        elif stage == "validate":
            self.val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)  # type: ignore
            if self.additional_dataset_classes is not None:
                self.val_list = [self.val]
                for data_dir, dataset_class, transform in zip(
                    self.additional_data_dirs, self.additional_dataset_classes, self.additional_transforms  # type: ignore
                ):
                    self.val_list.append(dataset_class(data_dir, partition="val", transform=transform))  # type: ignore
        elif stage == "predict":
            # TODO(liamvdv): delay until we know how things should look.
            # self.predict = None
            raise ValueError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("fit")
        return self.get_dataloader()(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> List[gtypes.BatchNletDataLoader]:
        self.setup("validate")
        dataloaders = [self.get_dataloader()(self.val, batch_size=self.batch_size, shuffle=False)]
        if self.additional_dataset_classes is not None:
            dataloaders.extend(
                [self.get_dataloader()(val, batch_size=self.batch_size, shuffle=False) for val in self.val_list[1:]]
            )
        return dataloaders

    def test_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("test")
        return self.get_dataloader()(self.test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("predict")
        # return self.get_dataloader()(self.predict, batch_size=self.batch_size, shuffle=False)
        raise NotImplementedError("predict_dataloader not implemented")

    def teardown(self, stage: str) -> None:
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass

    def get_num_classes(self, mode: Literal["train", "val", "test"]) -> int:  # HACK
        if mode == "train":
            train = self.dataset_class(
                self.data_dir,
                partition="train",
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )  # type: ignore
            return train.get_num_classes()  # type: ignore
        elif mode == "val":
            val_list = [self.dataset_class(self.data_dir, partition="val", transform=self.transforms)]  # type: ignore
            if self.additional_dataset_classes is not None:
                for data_dir, dataset_class, transform in zip(
                    self.additional_data_dirs, self.additional_dataset_classes, self.additional_transforms  # type: ignore
                ):
                    val_list.append(dataset_class(data_dir, partition="val", transform=transform))  # type: ignore
            return sum(val.get_num_classes() for val in val_list)  # type: ignore
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


class NLetKFoldDataModule(NletDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        dataset_class: Optional[Type[Dataset[Any]]] = None,
        transforms: Optional[gtypes.Transform] = None,
        training_transforms: Optional[gtypes.Transform] = None,
        val_fold: int = 0,
        k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir, batch_size, dataset_class, transforms, training_transforms, **kwargs)
        self.val_fold = val_fold
        self.k = k

    def setup(self, stage: str) -> None:
        assert self.dataset_class is not None, "dataset_class must be set before calling setup"
        logger.info(
            f"setup {stage} for Dataset {self.dataset_class.__name__} via Dataload {self.get_dataloader().__name__}"
        )

        if stage == "fit":
            self.train = self.dataset_class(
                self.data_dir,
                partition="train",
                val_i=self.val_fold,
                k=self.k,
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )  # type: ignore
            self.val = self.dataset_class(
                self.data_dir, partition="val", val_i=self.val_fold, k=self.k, transform=self.transforms
            )  # type: ignore

            if self.additional_dataset_classes is not None:
                self.val_list = [self.val]
                for data_dir, dataset_class, transform in zip(
                    self.additional_data_dirs, self.additional_dataset_classes, self.additional_transforms  # type: ignore
                ):
                    self.val_list.append(dataset_class(data_dir, partition="val", transform=transform))  # type: ignore
        elif stage == "test":
            self.test = self.dataset_class(
                self.data_dir, partition="test", val_i=self.val_fold, k=self.k, transform=self.transforms
            )  # type: ignore
        elif stage == "validate":
            self.val = self.dataset_class(
                self.data_dir, partition="val", val_i=self.val_fold, k=self.k, transform=self.transforms
            )  # type: ignore
            if self.additional_dataset_classes is not None:
                self.val_list = [self.val]
                for data_dir, dataset_class, transform in zip(
                    self.additional_data_dirs, self.additional_dataset_classes, self.additional_transforms  # type: ignore
                ):
                    self.val_list.append(dataset_class(data_dir, partition="val", transform=transform))  # type: ignore

        elif stage == "predict":
            # TODO(liamvdv): delay until we know how things should look.
            # self.predict = None
            raise ValueError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def get_num_classes(self, mode: Literal["train", "val", "test"]) -> int:  # HACK
        if mode == "train":
            train = self.dataset_class(
                self.data_dir,
                partition="train",
                val_i=self.val_fold,
                k=self.k,
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )  # type: ignore
            return train.get_num_classes()  # type: ignore
        elif mode == "val":
            val_list = [self.dataset_class(self.data_dir, partition="val", val_i=self.val_fold, k=self.k, transform=self.transforms)]  # type: ignore
            if self.additional_dataset_classes is not None:
                for data_dir, dataset_class, transform in zip(
                    self.additional_data_dirs, self.additional_dataset_classes, self.additional_transforms  # type: ignore
                ):
                    val_list.append(dataset_class(data_dir, partition="val", transform=transform))  # type: ignore
            return sum(val.get_num_classes() for val in val_list)  # type: ignore
        elif mode == "test":
            test = self.dataset_class(
                self.data_dir, partition="test", val_i=self.val_fold, k=self.k, transform=self.transforms
            )  # type: ignore
            return test.get_num_classes()  # type: ignore
        else:
            raise ValueError(f"unknown mode '{mode}'")


class TripletKFoldDataModule(NLetKFoldDataModule):
    def get_dataloader(self) -> Callable[[Dataset[Any], int, bool], gtypes.BatchTripletDataLoader]:
        return TripletDataLoader


class QuadletKFoldDataModule(NLetKFoldDataModule):
    def get_dataloader(self) -> Callable[[Dataset[Any], int, bool], gtypes.BatchQuadletDataLoader]:
        return QuadletDataLoader


class SimpleKFoldDataModule(NLetKFoldDataModule):
    def get_dataloader(self) -> Callable[[Dataset[Any], int, bool], gtypes.BatchSimpleDataLoader]:
        return SimpleDataLoader
