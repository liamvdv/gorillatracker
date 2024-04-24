from typing import Any, Callable, Tuple, Union

import cv2.typing as cvt
import torch

# Position top left, bottom right
BoundingBox = Tuple[Tuple[int, int], Tuple[int, int]]
Image = cvt.MatLike

Id = str
Label = Union[str, int]
TripletLabel = Tuple[Label, Label, Label]
TripletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BatchId = Tuple[Id, ...]
BatchLabel = Tuple[Label, ...]
BatchTripletIds = Tuple[BatchId, BatchId, BatchId]
BatchTripletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel]
BatchTripletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

LossPosNegDist = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BatchQuadletIds = Tuple[BatchId, BatchId, BatchId, BatchId]
BatchQuadletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel, BatchLabel]
BatchQuadletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

BatchTripletDataLoader = torch.utils.data.DataLoader[Tuple[BatchTripletIds, BatchTripletValue, BatchTripletLabel]]
BatchQuadletDataLoader = torch.utils.data.DataLoader[Tuple[BatchQuadletIds, BatchQuadletValue, BatchQuadletLabel]]
# BatchSimpleDataLoader = torch.utils.data.DataLoader[Tuple[torch.Tensor]], Tuple[BatchLabel]
BatchSimpleDataLoader = Any

BatchNletDataLoader = Union[BatchTripletDataLoader, BatchQuadletDataLoader]

TripletBatch = Tuple[BatchTripletIds, BatchTripletValue, BatchTripletLabel]
QuadletBatch = Tuple[BatchQuadletIds, BatchQuadletValue, BatchQuadletLabel]

NletBatch = Union[TripletBatch, QuadletBatch]

MergedLabels = Union[BatchLabel, torch.Tensor]


Transform = Callable[[Any], Any]
