import logging

import lightning as L
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import gorillatracker.ssl_pipeline.graph_builder as graph_builder
import gorillatracker.type_helper as gtypes
from gorillatracker.ssl_pipeline.dataset import SSLDataset, build_quadlet, build_triplet
from gorillatracker.transform_utils import SquarePad
from gorillatracker.train_utils import get_dataset_class

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
            train_graph = graph_builder.DEMO_get_clique_graph()
            self.train = SSLDataset(
                train_graph,
                build_triplet,
                "train",
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )
            self.val = self.train  # TODO(memben)
        elif stage == "test":
            self.test = None
            raise NotImplementedError("test not yet supported by data module.")
        elif stage == "validate":
            train_graph = graph_builder.DEMO_get_clique_graph()
            self.val = SSLDataset(
                train_graph,
                build_triplet,
                "train",
                transform=transforms.Compose([self.transforms, self.training_transforms]),
            )  # TODO(memben)
        elif stage == "predict":
            self.predict = None
            raise NotImplementedError("stage predict not yet supported by data module.")
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def train_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("fit")
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("validate")
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("test")
        raise NotImplementedError("test_dataloader not implemented")
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def predict_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("predict")
        raise NotImplementedError("predict_dataloader not implemented")

    def teardown(self, stage: str) -> None:
        # NOTE(liamvdv): used to clean-up when the run is finished
        pass

    def collate_fn(self, batch: list[gtypes.FlatNlet]) -> gtypes.NletBatch:
        nlets = tuple(self.collate_flat_nlet(flat_nlet) for flat_nlet in batch)

        ids_batch = tuple(nlet[0] for nlet in nlets)
        values_batch = tuple(torch.stack(nlet[1]) for nlet in nlets)
        labels_batch = tuple(nlet[2] for nlet in nlets)

        return ids_batch, values_batch, labels_batch

    def collate_flat_nlet(self, flat_nlet: gtypes.FlatNlet) -> gtypes.Nlet:
        ids = tuple(image_label[0] for image_label in flat_nlet)
        images = tuple(image_label[1] for image_label in flat_nlet)
        labels = tuple(image_label[2] for image_label in flat_nlet)
        image_tensors = tuple(torch.tensor(np.array(image)) for image in images)
        return ids, image_tensors, labels

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
