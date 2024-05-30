from typing import Any, Callable, Tuple, Union

import cv2.typing as cvt
import torch
from torch.utils.data import DataLoader

# Position top left, bottom right
BoundingBox = Tuple[Tuple[int, int], Tuple[int, int]]
Image = cvt.MatLike

Id = str
Label = Union[str, int]

NletId = Tuple[Id, ...]  # e.g (anchor_id, positive_id, negative_id)
NletLabel = Tuple[Label, ...]  # e.g (anchor_label, positive_label, negative_label)
NletValue = Tuple[torch.Tensor, ...]  # e.g (anchor_image, positive_image, negative_image)
Nlet = Tuple[NletId, NletValue, NletLabel]


# NOTE(memben): Concrete type hints are all wrong
BatchId = Tuple[Id, ...]
BatchLabel = Tuple[Label, ...]
BatchTripletIds = Tuple[BatchId, BatchId, BatchId]
BatchTripletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel]
BatchTripletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

LossPosNegDist = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BatchQuadletIds = Tuple[BatchId, BatchId, BatchId, BatchId]
BatchQuadletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel, BatchLabel]
BatchQuadletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

TripletBatch = Tuple[BatchTripletIds, BatchTripletValue, BatchTripletLabel]
QuadletBatch = Tuple[BatchQuadletIds, BatchQuadletValue, BatchQuadletLabel]
# WRONG ensd here

NletBatchIds = Tuple[NletId, ...]  # e.g. ((anchor_id_1, positive_id_1, negative_id_1), ...)
NletBatchValues = Tuple[NletValue, ...]  # e.g. ((anchor_image_1, positive_image_1, negative_image_1), ...)
NletBatchLabels = Tuple[NletLabel, ...]  # e.g. ((anchor_label_1, positive_label_1, negative_label_1), ...)

NletBatch = Tuple[NletBatchIds, NletBatchValues, NletBatchLabels]

BatchTripletDataLoader = DataLoader[TripletBatch]
BatchQuadletDataLoader = DataLoader[QuadletBatch]
# BatchSimpleDataLoader = torch.utils.data.DataLoader[Tuple[torch.Tensor]], Tuple[BatchLabel]
BatchSimpleDataLoader = Any

BatchNletDataLoader = DataLoader[NletBatch]


MergedLabels = Union[BatchLabel, torch.Tensor]


Transform = Callable[[Any], Any]
