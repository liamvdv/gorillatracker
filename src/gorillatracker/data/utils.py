from itertools import chain

import torch

from gorillatracker.type_helper import FlatNletBatch, NletBatch


def lazy_batch_size(batch: NletBatch) -> int:
    anchor_ids = batch[0][0]
    return len(anchor_ids)


def flatten_batch(batch: NletBatch) -> FlatNletBatch:
    ids, images, labels = batch
    # transform ((a1, a2), (p1, p2), (n1, n2)) to (a1, a2, p1, p2, n1, n2)
    flat_ids = tuple(chain.from_iterable(ids))
    # transform ((a1, a2), (p1, p2), (n1, n2)) to (a1, a2, p1, p2, n1, n2)
    flat_labels = torch.flatten(torch.tensor(labels))
    # transform ((a1: Tensor, a2: Tensor), (p1: Tensor, p2: Tensor), (n1: Tensor, n2: Tensor))  to (a1, a2, p1, p2, n1, n2)
    flat_images = torch.stack(list(chain.from_iterable(images)), dim=0)
    return flat_ids, flat_images, flat_labels
