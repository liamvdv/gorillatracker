from torchvision.transforms import Compose, Grayscale, ToTensor

from gorillatracker.data_modules.data_modules import QuadletDataModule, TripletDataModule
from gorillatracker.datasets.mnist import MNISTDataset


def get_mnist_transforms():
    return Compose(
        [
            Grayscale(num_output_channels=3),
            ToTensor(),
        ]
    )


class MNISTTripletDataModule(TripletDataModule):
    def get_dataset_class(self):
        return MNISTDataset

    def get_transforms(self):
        return get_mnist_transforms()


class MNISTQuadletDataModule(QuadletDataModule):
    def get_dataset_class(self):
        return MNISTDataset

    def get_transforms(self):
        return get_mnist_transforms()
