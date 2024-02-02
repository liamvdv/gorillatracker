import itertools
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import gorillatracker.type_helper as gtypes

T = TypeVar("T")
R = TypeVar("R")

LabelSection = DefaultDict[Union[int, str], Tuple[int, int]]


def generate_labelsection(sorted_value_labels: Sequence[Tuple[Any, Union[int, str]]]) -> LabelSection:
    n = len(sorted_value_labels)
    labelsection: LabelSection = defaultdict(lambda: (-1, -1))
    prev_label = None
    prev_start = None
    for i, (_, label) in enumerate(sorted_value_labels):
        assert prev_label is None or prev_label <= label, "dataset passed to TripletSampler must be label-sorted."
        if prev_label is None:
            prev_label = label
            prev_start = i
        elif prev_label != label:
            labelsection[prev_label] = (prev_start, i - prev_start)
            prev_label = label
            prev_start = i
    assert prev_label is not None and prev_start is not None  # make typing happy
    if prev_label:
        labelsection[prev_label] = (prev_start, n - prev_start)
    return labelsection


def iter_index_permutations_generator(n: int) -> Iterator[Tuple[int, ...]]:
    """
    returns all possibel index permutations in a systematic order.
    WARN(liamvdv): the systematic order makes the first indices occure predominantly.
                   This screws the batch sampling towards lower labels (since dataset
                   is passed sorted).
    Behaviour Example:
        (0, 1, 2)
        (0, 2, 1)
        (1, 0, 2)
        (1, 2, 0)
        (2, 0, 1)
        (2, 1, 0)
    """
    return itertools.permutations(list(range(n)))


def index_permuation_generator(n: int) -> Generator[List[int], None, None]:
    """
    returns a generator that returns a random index permutation. Does not ensure
    all permuations are seen. Reproducable via torch.seed.
    """
    while True:
        yield torch.randperm(n).tolist()


def randint_except(start: int, end: int, excluded: int) -> int:
    """[start, end)"""
    size = end - start
    if size == 0:
        raise ValueError("class must have at least 1 instance")
    elif size == 1:
        # TODO(liamvdv): filter out 1 instances?
        # raise ValueError("class must have >=2 instances")
        return start
    while True:
        idx: int = torch.randint(start, end, (1,)).item()  # type: ignore
        if idx != excluded:
            return idx


class ToNthDataset(Dataset[Tuple[Tuple[T, ...], Tuple[R, ...]]], Generic[T, R]):
    """
    ToNthDataset allows N index accesses at once on a single index access Dataset.
    """

    def __init__(self, dataset: Dataset[Tuple[T, R]], transform: gtypes.Transform = lambda x: x) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idxs: Union[int, List[int], torch.Tensor]) -> Tuple[Tuple[T, ...], Tuple[R, ...]]:
        if torch.is_tensor(idxs):  # type: ignore
            idxs = idxs.tolist()  # type: ignore
        if isinstance(idxs, int):
            idxs = [idxs]

        xs, ys = [], []
        for idx in idxs:
            x, y = self.dataset[idx]
            x = self.transform(x)
            xs.append(x)
            ys.append(y)

        return tuple(xs), tuple(ys)


class TripletSampler(Sampler[Tuple[int, int, int]]):
    """Do not use DataLoader(..., shuffle=True) with TripletSampler."""

    def __init__(
        self,
        sorted_dataset: Sequence[Tuple[Any, gtypes.Label]],
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
            anchor_label = self.dataset[anchor][1]
            astart, alength = self.labelsection[anchor_label]
            positive = randint_except(astart, astart + alength, anchor)
            negative = self.any_sample_not(anchor_label)
            yield anchor, positive, negative


class QuadletSampler(Sampler[Tuple[int, int, int, int]]):
    """Do not use DataLoader(..., shuffle=True) with QuadletSampler."""

    def __init__(
        self,
        sorted_dataset: Sequence[Tuple[Any, gtypes.Label]],
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

    def __iter__(self) -> Iterator[Tuple[int, int, int, int]]:
        anchor_shuffle = next(self.shuffled_indices_generator)
        for anchor_positive in anchor_shuffle:
            anchor_p_label = self.dataset[anchor_positive][1]
            pstart, plength = self.labelsection[anchor_p_label]
            positive = randint_except(pstart, pstart + plength, anchor_positive)

            anchor_negative = self.any_sample_not(anchor_p_label)
            negative_label = self.dataset[anchor_negative][1]
            nstart, nlength = self.labelsection[negative_label]
            negative = randint_except(nstart, nstart + nlength, anchor_negative)
            yield anchor_positive, positive, anchor_negative, negative


class FreezeSampler(Sampler[T]):
    """
    FreezeSampler is a wrapper around a Sampler that freezes the indices returned by
    the first iteration of the sampler and returns those in every subsequent iteration.
    """

    def __init__(self, sampler: Sampler[T]) -> None:
        self.sampler = sampler
        self.cache: Optional[List[T]] = None

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore

    def __iter__(self) -> Iterator[T]:
        if self.cache is None:
            self.cache = list(self.sampler)
        assert self.cache is not None
        return iter(self.cache)


def TripletDataLoader(
    dataset: Dataset[Tuple[Any, gtypes.Label]], batch_size: int, shuffle: bool = True
) -> gtypes.BatchTripletDataLoader:
    """
    TripletDataLoader will take any Dataset that returns a single sample in the form of
    (value, label) on __getitem__ and transform it into an efficient Triplet DataLoader.
    If shuffle=True, the dataset will be shuffled on every epoch. If shuffle=False, the
    dataset will be shuffled once at the start and not after that.
    """
    label_sorted_dataset = sorted(dataset, key=lambda t: t[1])  # type: ignore
    sampler = TripletSampler(label_sorted_dataset)
    if not shuffle:
        sampler = FreezeSampler(sampler)  # type: ignore
    final_dataset = ToNthDataset(label_sorted_dataset)
    return DataLoader(final_dataset, sampler=sampler, shuffle=False, batch_size=batch_size)


def QuadletDataLoader(
    dataset: Dataset[Tuple[Any, gtypes.Label]], batch_size: int, shuffle: bool = True
) -> gtypes.BatchQuadletDataLoader:
    """
    QuadletDataLoader will take any Dataset that returns a single sample in the form of
    (value, label) on __getitem__ and transform it into an efficient Quadlet DataLoader.
    If shuffle=True, the dataset will be shuffled on every epoch. If shuffle=False, the
    dataset will be shuffled once at the start and not after that.
    """
    label_sorted_dataset = sorted(dataset, key=lambda t: t[1])  # type: ignore
    sampler = QuadletSampler(label_sorted_dataset)
    if not shuffle:
        sampler = FreezeSampler(sampler)  # type: ignore
    final_dataset = ToNthDataset(label_sorted_dataset)
    return DataLoader(final_dataset, sampler=sampler, shuffle=False, batch_size=batch_size)


def SimpleDataLoader(
    dataset: Dataset[Tuple[Any, gtypes.Label]], batch_size: int, shuffle: bool = True
) -> gtypes.BatchSimpleDataLoader:
    final_dataset = ToNthDataset(dataset)
    return DataLoader(dataset=final_dataset, shuffle=shuffle, batch_size=batch_size)
