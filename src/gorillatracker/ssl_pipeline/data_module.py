import logging
from typing import Any, Literal, Optional, Type

import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import gorillatracker.type_helper as gtypes
from gorillatracker.type_helper import BatchTripletDataLoader
from torch.utils.data import DataLoader, Dataset, Sampler

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSLDataModule(L.LightningDataModule):
    """
    Base class for triplet/quadlet data modules, implementing shared functionality.
    """

    def __init__(
        self,
        batch_size: int = 32,
        transforms: Optional[gtypes.Transform] = None,
        training_transforms: Optional[gtypes.Transform] = None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.training_transforms = training_transforms
        self.batch_size = batch_size


    def setup(self, stage: str) -> None:
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
        self.setup("fit")
        return self.get_dataloader()(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> gtypes.BatchNletDataLoader:
        self.setup("validate")
        return self.get_dataloader()(self.val, batch_size=self.batch_size, shuffle=False)

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

class TripletSampler(Sampler[tuple[int, int, int]]):
    """Do not use DataLoader(..., shuffle=True) with TripletSampler."""

    def __init__(
        self,
        sorted_dataset: Sequence[Tuple[gtypes.Id, Any, gtypes.Label]],
        shuffled_indices_generator: Callable[[int], Generator[List[int], None, None]] = index_permuation_generator,
    ):
        self.dataset = sorted_dataset
        self.n = len(self.dataset)
        self.labelsection = generate_labelsection(self.dataset)
        self.shuffled_indices_generator = shuffled_indices_generator(self.n)

    def any_sample_not(self, label: gtypes.Label) -> int:
        start, length = self.labelsection[label]
        end = start + length
        i: int = torch.randint(self.n - length, (1,)).item()  # type: ignore
        if start <= i and i < end:
            i += length
        return i

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        anchor_shuffle = next(self.shuffled_indices_generator)
        for anchor in anchor_shuffle:
            anchor_label = self.dataset[anchor][2]
            astart, alength = self.labelsection[anchor_label]
            positive = randint_except(astart, astart + alength, anchor)
            negative = self.any_sample_not(anchor_label)
            yield anchor, positive, negative