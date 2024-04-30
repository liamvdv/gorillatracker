from typing import Any, Callable, Tuple, Union

import cv2.typing as cvt
import torch
from torch.utils.data import DataLoader

# Position top left, bottom right
BoundingBox = Tuple[Tuple[int, int], Tuple[int, int]]
Image = cvt.MatLike

Id = str
Label = Union[str, int]

NletId = Tuple[Id, ...]
NletLabel = Tuple[Label, ...]
NletValue = Tuple[torch.Tensor, ...]
Nlet = Tuple[NletId, NletValue, NletLabel]


BatchId = Tuple[Id, ...]
BatchLabel = Tuple[Label, ...]
BatchTripletIds = Tuple[BatchId, BatchId, BatchId]
BatchTripletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel]
# stacked tensors
BatchTripletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

LossPosNegDist = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BatchQuadletIds = Tuple[BatchId, BatchId, BatchId, BatchId]
BatchQuadletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel, BatchLabel]
BatchQuadletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

TripletBatch = Tuple[BatchTripletIds, BatchTripletValue, BatchTripletLabel]
QuadletBatch = Tuple[BatchQuadletIds, BatchQuadletValue, BatchQuadletLabel]

NletBatchIds = Tuple[BatchId, ...]
NletBatchLabels = Tuple[BatchLabel, ...]
NletBatchValues = Tuple[torch.Tensor, ...]

NletBatch = Tuple[NletBatchIds, NletBatchValues, NletBatchLabels]

BatchTripletDataLoader = DataLoader[TripletBatch]
BatchQuadletDataLoader = DataLoader[QuadletBatch]
# BatchSimpleDataLoader = torch.utils.data.DataLoader[Tuple[torch.Tensor]], Tuple[BatchLabel]
BatchSimpleDataLoader = Any

BatchNletDataLoader = DataLoader[NletBatch]


MergedLabels = Union[BatchLabel, torch.Tensor]


Transform = Callable[[Any], Any]
