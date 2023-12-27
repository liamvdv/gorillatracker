import math
from typing import Tuple

import torch

import gorillatracker.type_helper as gtypes

# import variational prototype learning from insightface


eps = 1e-16  # an arbitrary small value to be used for numerical stability tricks


class ArcFace(torch.nn.Module):  # TODO (rob2u): test
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(self, embedding_size, num_classes, s=64.0, margin=0.5, *args, **kwargs):
        super(ArcFace, self).__init__(*args, **kwargs)
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.prototypes = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        torch.nn.init.xavier_uniform_(self.prototypes)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> gtypes.LossPosNegDist:
        """Forward pass of the ArcFace loss function"""

        # get cos(theta) for each embedding and prototype
        cos_theta = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings), torch.nn.functional.normalize(self.prototypes)
        )
        sine_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2)).clamp(eps, 1.0 - eps)
        phi = (
            cos_theta * self.cos_m - sine_theta * self.sin_m
        )  # additionstheorem cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

        mask = torch.zeros(cos_theta.size(), device=cos_theta.device)
        mask.scatter_(1, labels.view(-1, 1).long(), 1)  # mask is one-hot encoded labels

        output = (mask * phi) + ((1.0 - mask) * cos_theta)  # NOTE: sometimes there is an additional penalty term
        output *= self.s
        loss = self.ce(output, labels)

        return loss, torch.Tensor([-1.0]), torch.Tensor([-1.0])  # dummy values for pos/neg distances


class VariationalPrototypeLearning(
    torch.nn.Module
):  # TODO (rob2u): test NOTE: this is not the completely original implementation
    """Variational Prototype Learning Loss
    See https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Variational_Prototype_Learning_for_Deep_Face_Recognition_CVPR_2021_paper.pdf
    """

    def __init__(
        self,
        embedding_size,
        num_classes,
        batch_size,
        s=64.0,
        margin=0.5,
        delta_t=100,
        lambda_membank=0.001,
        *args,
        **kwargs,
    ) -> None:
        super(VariationalPrototypeLearning, self).__init__(*args, **kwargs)
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.delta_t = delta_t
        self.lambda_membank = lambda_membank
        self.prototypes = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        torch.nn.init.xavier_uniform_(self.prototypes)
        self.ce = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        self.memory_bank_ptr = 0  # pointer to the current memory bank position that will be replaced
        self.memory_bank = torch.zeros(delta_t * batch_size, embedding_size)
        self.memory_bank_labels = torch.zeros(delta_t * batch_size, dtype=torch.int32)
        self.using_memory_bank = False

    def update_memory_bank(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Updates the memory bank with the current batch of embeddings and labels"""
        self.memory_bank[
            self.memory_bank_ptr * self.batch_size : (self.memory_bank_ptr + 1) * self.batch_size
        ] = embeddings
        self.memory_bank_labels[
            self.memory_bank_ptr * self.batch_size : (self.memory_bank_ptr + 1) * self.batch_size
        ] = labels
        self.memory_bank_ptr = (self.memory_bank_ptr + 1) % self.delta_t

    @torch.no_grad()
    def get_memory_bank_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the prototypes and their frequency in the memory bank"""

        prototypes = torch.zeros(self.num_classes, self.embedding_size)
        frequency = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            prototypes[i] = torch.mean(self.memory_bank[self.memory_bank_labels == i], dim=0)
            frequency[i] = torch.sum(self.memory_bank_labels == i)
        return prototypes, frequency

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> gtypes.LossPosNegDist:
        """Forward pass of the Variational Prototype Learning loss function"""
        if self.using_memory_bank:
            self.update_memory_bank(embeddings, labels)
            mem_bank_prototypes, prototype_frequency = self.get_memory_bank_prototypes()
            relative_frequency = prototype_frequency / torch.sum(prototype_frequency)
            prototypes = (
                1 - self.lambda_membank * relative_frequency
            ) * self.prototypes + self.lambda_membank * relative_frequency * mem_bank_prototypes
        else:
            prototypes = self.prototypes

        cos_theta = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings), torch.nn.functional.normalize(prototypes)
        )
        sine_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2)).clamp(eps, 1.0 - eps)
        phi = (
            cos_theta * self.cos_m - sine_theta * self.sin_m
        )  # additionstheorem cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

        mask = torch.zeros(cos_theta.size(), device=cos_theta.device)
        mask.scatter_(1, labels.view(-1, 1).long(), 1)  # mask is one-hot encoded labels

        output = (mask * phi) + ((1.0 - mask) * cos_theta)  # NOTE: sometimes there is an additional penalty term
        output *= self.s
        loss = self.ce(output, labels)

        return loss, torch.Tensor([-1.0]), torch.Tensor([-1.0])  # dummy values for pos/neg distances
