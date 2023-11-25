from typing import Union

import torchvision.transforms as transforms

from gorillatracker.data_modules.data_modules import QuadletDataModule, TripletDataModule
from gorillatracker.data_modules.transform_utils import SquarePad
from gorillatracker.datasets.cxl import CXLDataset

Label = Union[int, str]


def get_cxl_transforms():
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )


class CXLTripletDataModule(TripletDataModule):
    def get_dataset_class(self):
        return CXLDataset

    def get_transforms(self):
        return get_cxl_transforms()


class CXLQuadletDataModule(QuadletDataModule):
    def get_dataset_class(self):
        return CXLDataset

    def get_transforms(self):
        return get_cxl_transforms()


if __name__ == "__main__":
    for dm in [CXLTripletDataModule, CXLQuadletDataModule]:
        mnist_dm = dm(
            "data/splits/ground_truth-cxl-closedset--mintraincount-3-seed-42-train-70-val-15-test-15",
        )
        mnist_dm.setup("test")
        for v in mnist_dm.test_dataloader():
            print(v)
