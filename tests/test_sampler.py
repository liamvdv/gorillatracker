import string

from gorillatracker.data_loaders import TripletDataLoader, TripletSampler


def test_triplet_sampler() -> None:
    epochs = 100
    n_individuals = 4
    dataset = sorted(enumerate(string.ascii_lowercase * n_individuals), key=lambda t: t[1])
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
    epochs = 100
    n_individuals = 4
    dataset = list(enumerate(string.ascii_lowercase * n_individuals))
    dl = TripletDataLoader(dataset, batch_size=1)  # type: ignore
    epoch_batches = []
    for _ in range(epochs):
        epoch_batches.append([v for v in dl])

    for i in range(len(epoch_batches)):
        for j in range(len(epoch_batches)):
            if i != j:
                # NOTE(liamvdv): does deep comparison work
                assert epoch_batches[i] != epoch_batches[j]
