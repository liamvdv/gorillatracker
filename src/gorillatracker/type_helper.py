from typing import Any, Callable, Tuple, Union

import cv2.typing as cvt
import torch

BoundingBox = Tuple[Tuple[int, int], Tuple[int, int]]
Image = cvt.MatLike

Label = Union[str, int]
TripletLabel = Tuple[Label, Label, Label]
TripletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BatchLabel = Tuple[Label]
BatchTripletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel]
BatchTripletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

BatchQuadletLabel = Tuple[BatchLabel, BatchLabel, BatchLabel, BatchLabel]
BatchQuadletValue = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

BatchTripletDataLoader = torch.utils.data.DataLoader[Tuple[BatchTripletValue, BatchTripletLabel]]
BatchQuadletDataLoader = torch.utils.data.DataLoader[Tuple[BatchQuadletValue, BatchQuadletLabel]]

BatchNletDataLoader = Union[BatchTripletDataLoader, BatchQuadletDataLoader]

TripletBatch = Tuple[BatchTripletValue, BatchTripletLabel]
QuadletBatch = Tuple[BatchQuadletValue, BatchQuadletLabel]

NletBatch = Union[TripletBatch, QuadletBatch]

MergedLabels = Union[BatchLabel, torch.Tensor]


Transform = Callable[[Any], Any]
