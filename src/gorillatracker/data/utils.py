from itertools import chain

import torch

from gorillatracker.type_helper import FlatNletBatch, NletBatch


def lazy_batch_size(batch: NletBatch) -> int:
    anchor_ids = batch[0][0]
    return len(anchor_ids)


def flatten_batch(batch: NletBatch) -> FlatNletBatch:
    """Returns one tuple for ids, images, labels. Each id/image/label has size N * batch size,
    where N is from Nlet. E.g. for Quadlets, it would be `[anchors, positives, negatives, anchor_negative]`."""
    ids, images, labels = batch
    # transform ((a1, a2), (p1, p2), (n1, n2)) to (a1, a2, p1, p2, n1, n2)
    flat_ids = tuple(chain.from_iterable(ids))
    # transform ((a1, a2), (p1, p2), (n1, n2)) to (a1, a2, p1, p2, n1, n2)
    flat_labels = torch.cat(labels)
    # transform ((a1: Tensor, a2: Tensor), (p1: Tensor, p2: Tensor), (n1: Tensor, n2: Tensor))  to (a1, a2, p1, p2, n1, n2)
    flat_images = torch.cat(images)
    return flat_ids, flat_images, flat_labels
