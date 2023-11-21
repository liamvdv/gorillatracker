import logging

import lightning as L

from gorillatracker.data_loaders import QuadletDataLoader, TripletDataLoader

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NletDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, dataset_class=None, transforms=None):
        super().__init__()
        self.transforms = transforms or self.get_transforms()
        self.dataset_class = dataset_class or self.get_dataset_class()
        self.data_dir = data_dir
        self.batch_size = batch_size

    @classmethod
    def from_training_args(cls, args):
        return cls(data_dir=args.data_dir, batch_size=args.batch_size)

    def get_transforms(self):
        return None

    def get_dataset_class(self):
        raise Exception("must provide a dataset_cls argument or subclass")

    def get_dataloader(self):
        raise Exception("logic error, ask liamvdv")

    def setup(self, stage: str):
        logger.info(f"setup {stage} for Dataset {self.dataset_class.__name__} via Dataload {self.get_dataloader().__name__}")

        if stage == "fit":
            self.train = self.dataset_class(self.data_dir, partition="train", transform=self.transforms)
            self.val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)
        elif stage == "test":
            self.test = self.dataset_class(self.data_dir, partition="test", transform=self.transforms)
        elif stage == "validate":
            self.val = self.dataset_class(self.data_dir, partition="val", transform=self.transforms)
        elif stage == "predict":
            # TODO(liamvdv): delay until we know how things should look.
            self.predict = None
            raise ValueError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self):
        return self.get_dataloader()(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return self.get_dataloader()(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return self.get_dataloader()(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.get_dataloader()(self.predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass


class TripletDataModule(NletDataModule):
    def get_dataloader(self):
        return TripletDataLoader


class QuadletDataModule(NletDataModule):
    def get_dataloader(self):
        return QuadletDataLoader
