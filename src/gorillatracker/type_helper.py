from typing import Callable, Tuple, Union

import cv2.typing as cvt
import torch
from PIL.Image import Image as PILImage
from torch.utils.data import DataLoader

# Position top left, bottom right
BoundingBox = Tuple[Tuple[int, int], Tuple[int, int]]

Image = cvt.MatLike

Id = str
Label = int

NletId = Tuple[Id, ...]  # e.g (anchor_id, positive_id, negative_id)
NletLabel = Tuple[Label, ...]  # e.g (anchor_label, positive_label, negative_label)
NletValue = Tuple[torch.Tensor, ...]  # e.g (anchor_image, positive_image, negative_image)
Nlet = Tuple[NletId, NletValue, NletLabel]


# NOTE(memben): Concrete type hints are all wrong
BatchId = Tuple[Id, ...]
BatchLabel = Tuple[Label, ...]

LossPosNegDist = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# WRONG ensd here

NletBatchIds = Tuple[
    Tuple[Id, ...], ...
]  # e.g. ((anchor_id_1, anchor_id2, ...), (positive_id_1, ...), (negative_id_1, ...), ...)
NletBatchValues = Tuple[
    Tuple[torch.Tensor, ...], ...
]  # e.g. ((anchor_image_1, anchor_image_2, ...), (positive_image_1, ...), (negative_image_1, ...), ...)
NletBatchLabels = Tuple[
    Tuple[Label, ...], ...
]  # e.g. ((anchor_label_1, anchor_label_2, ...), (positive_label_1, ...), (negative_label_1, ...), ...)

NletBatch = Tuple[NletBatchIds, NletBatchValues, NletBatchLabels]

FlatNletBatchIds = tuple[Id, ...]  # e.g. (anchor_id_1, ..., positive_id_1, ..., negative_id_1, ...)
FlatNletBatchValues = torch.Tensor  # e.g. (anchor_image_1, ..., positive_image_1, ..., negative_image_1, ...)
FlatNletBatchLabels = torch.Tensor  # e.g. (anchor_label_1, ..., positive_label_1, ..., negative_label_1, ...)

FlatNletBatch = Tuple[FlatNletBatchIds, FlatNletBatchValues, FlatNletBatchLabels]


# TODO(memben)
BatchNletDataLoader = DataLoader[NletBatch]


MergedLabels = Union[BatchLabel, torch.Tensor]


Transform = Callable[[PILImage], torch.Tensor]
TensorTransform = Callable[[torch.Tensor], torch.Tensor]
