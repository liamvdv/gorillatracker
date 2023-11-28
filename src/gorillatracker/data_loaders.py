import itertools
from collections import defaultdict
from typing import Any, List, Tuple

import torch
from torch.utils.data.dataloader import DataLoader, Sampler


def generate_labelsection(sorted_value_labels: List[Tuple[Any, Any]]):
    n = len(sorted_value_labels)
    labelsection = defaultdict(lambda: (None, None))
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
    if prev_label:
        labelsection[prev_label] = (prev_start, n - prev_start)
    return labelsection


def iter_index_permutations_generator(n: int):
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


def index_permuation_generator(n: int):
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
        idx = torch.randint(start, end, (1,)).item()
        if idx != excluded:
            return idx


class ToNthDataset:
    """
    ToNthDataset allows N index accesses at once on a single index access Dataset.
    """

    def __init__(self, dataset, transform=lambda x: x):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idxs):
        if torch.is_tensor(idxs):
            idxs = idxs.tolist()

        xs, ys = [], []
        for idx in idxs:
            x, y = self.dataset[idx]
            x = self.transform(x)
            xs.append(x)
            ys.append(y)

        return tuple(xs), tuple(ys)


class TripletSampler(Sampler):
    """Do not use DataLoader(..., shuffle=True) with TripletSampler."""

    def __init__(self, sorted_dataset, shuffled_indices_generator=index_permuation_generator):
        self.dataset = sorted_dataset
        self.n = len(self.dataset)
        self.labelsection = generate_labelsection(self.dataset)
        self.shuffled_indices_generator = shuffled_indices_generator(self.n)

    def any_sample_not(self, label) -> int:
        start, length = self.labelsection[label]
        end = start + length
        i = torch.randint(self.n - length, (1,)).item()
        if start <= i and i < end:
            i += length
        return i

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        anchor_shuffle = next(self.shuffled_indices_generator)
        for anchor in anchor_shuffle:
            anchor_label = self.dataset[anchor][1]
            astart, alength = self.labelsection[anchor_label]
            positive = randint_except(astart, astart + alength, anchor)
            negative = self.any_sample_not(anchor_label)
            yield anchor, positive, negative


class QuadletSampler(TripletSampler):
    """Do not use DataLoader(..., shuffle=True) with QuadletSampler."""

    def __iter__(self):
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


class FreezeSampler(Sampler):
    """
    FreezeSampler is a wrapper around a Sampler that freezes the indices returned by
    the first iteration of the sampler and returns those in every subsequent iteration.
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.cache = None

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        if self.cache is None:
            self.cache = list(self.sampler)
        return iter(self.cache)


def TripletDataLoader(dataset, batch_size, shuffle=True):
    """
    TripletDataLoader will take any Dataset that returns a single sample in the form of
    (value, label) on __getitem__ and transform it into an efficient Triplet DataLoader.
    If shuffle=True, the dataset will be shuffled on every epoch. If shuffle=False, the
    dataset will be shuffled once at the start and not after that.
    """
    label_sorted_dataset = sorted(dataset, key=lambda t: t[1])
    sampler = TripletSampler(label_sorted_dataset)
    if not shuffle:
        sampler = FreezeSampler(sampler)
    final_dataset = ToNthDataset(label_sorted_dataset)
    return DataLoader(final_dataset, sampler=sampler, shuffle=False, batch_size=batch_size)


def QuadletDataLoader(dataset, batch_size, shuffle=True):
    """
    QuadletDataLoader will take any Dataset that returns a single sample in the form of
    (value, label) on __getitem__ and transform it into an efficient Quadlet DataLoader.
    If shuffle=True, the dataset will be shuffled on every epoch. If shuffle=False, the
    dataset will be shuffled once at the start and not after that.
    """
    label_sorted_dataset = sorted(dataset, key=lambda t: t[1])
    sampler = QuadletSampler(label_sorted_dataset)
    if not shuffle:
        sampler = FreezeSampler(sampler)
    final_dataset = ToNthDataset(label_sorted_dataset)
    return DataLoader(final_dataset, sampler=sampler, shuffle=False, batch_size=batch_size)
