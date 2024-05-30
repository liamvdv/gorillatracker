import string
from typing import List, Tuple

import numpy as np
from torch import Tensor

from gorillatracker.data_loaders import TripletDataLoader, TripletSampler
from gorillatracker.type_helper import Id


def generate_fake_dataset(
    n_individuals: int = 4, all_labels: List[str] = list(string.ascii_lowercase)
) -> List[Tuple[Id, Tensor, str]]:
    n = len(all_labels) * n_individuals
    labels = all_labels * n_individuals
    ids = list(map(str, range(n)))
    samples = [Tensor(np.eye(192 * 2 * 3))] * n
    dataset = sorted(zip(ids, samples, labels), key=lambda x: x[2])
    return dataset


def test_triplet_sampler() -> None:
    epochs = 100
    dataset = generate_fake_dataset()
    sampler = TripletSampler(dataset)

    all_different = []
    for _ in range(epochs):
        epoch_samples = []
        for sampler_output in sampler:
            anchor, positive, negative = sampler_output
            epoch_samples.append(sampler_output)
        all_different.append(epoch_samples)

    for i in range(len(all_different)):
        for j in range(len(all_different)):
            if i != j:
                # NOTE(liamvdv): does deep comparison work
                assert all_different[i] != all_different[j]


def test_data_loader() -> None:
    epochs = 10
    dataset = generate_fake_dataset()
    dl = TripletDataLoader(dataset, batch_size=1)  # type: ignore
    epoch_batches = []
    for _ in range(epochs):
        epoch_batches.append([v for v in dl])

    for i in range(len(epoch_batches)):
        for j in range(len(epoch_batches)):
            if i != j:
                # NOTE(liamvdv): does deep comparison work
                assert epoch_batches[i] != epoch_batches[j]
