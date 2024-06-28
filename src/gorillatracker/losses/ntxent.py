from typing import Any

import torch
from lightly.loss import NTXentLoss as NTXent
from torch import nn

import gorillatracker.type_helper as gtypes


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.loss = NTXent(temperature=temperature)

    def forward(self, embeddings: torch.Tensor, labels: gtypes.MergedLabels, **kwargs: Any) -> gtypes.LossPosNegDist:
        half = embeddings.size(0) // 2
        anchors, positives = embeddings[:half], embeddings[half:]
        NO_VALUE = torch.tensor([-1])
        return self.loss(anchors, positives), NO_VALUE, NO_VALUE
