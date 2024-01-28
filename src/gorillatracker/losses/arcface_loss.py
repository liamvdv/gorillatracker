import math
from typing import Any, Tuple

import torch

import gorillatracker.type_helper as gtypes

# import variational prototype learning from insightface


eps = 1e-16  # an arbitrary small value to be used for numerical stability


class ArcFaceLoss(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        s: float = 64.0,
        margin: float = 0.5,
        accelerator: str = "cpu",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(ArcFaceLoss, self).__init__(*args, **kwargs)
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.num_classes = num_classes
        if accelerator == "cuda":
            self.prototypes = torch.nn.Parameter(torch.cuda.FloatTensor(num_classes, embedding_size))  # type: ignore
        else:
            self.prototypes = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_size))

        torch.nn.init.xavier_uniform_(self.prototypes)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> gtypes.LossPosNegDist:
        """Forward pass of the ArcFace loss function"""

        # get cos(theta) for each embedding and prototype
        prototypes = self.prototypes.to(embeddings.device)

        if labels.device != embeddings.device:
            labels.to(embeddings.device)

        cos_theta = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings), torch.nn.functional.normalize(prototypes)
        )
        sine_theta = torch.sqrt(
            torch.maximum(1.0 - torch.pow(cos_theta, 2), torch.tensor([eps], device=cos_theta.device))
        ).clamp(eps, 1.0 - eps)
        phi = (
            cos_theta * self.cos_m - sine_theta * self.sin_m
        )  # additionstheorem cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

        mask = torch.zeros(cos_theta.size(), device=cos_theta.device)
        mask.scatter_(1, labels.view(-1, 1).long(), 1)  # mask is one-hot encoded labels

        output = (mask * phi) + ((1.0 - mask) * cos_theta)  # NOTE: sometimes there is an additional penalty term
        output *= self.s
        loss = self.ce(output, labels)

        return loss, torch.Tensor([-1.0]), torch.Tensor([-1.0])  # dummy values for pos/neg distances

    def set_weights(self, weights: torch.Tensor) -> None:
        """Sets the weights of the prototypes"""
        assert weights.shape == self.prototypes.shape

        if torch.cuda.is_available() and self.prototypes.device != weights.device:
            weights = weights.cuda()

        self.prototypes = torch.nn.Parameter(weights)


class VariationalPrototypeLearning(torch.nn.Module):  # NOTE: this is not the completely original implementation
    """Variational Prototype Learning Loss
    See https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Variational_Prototype_Learning_for_Deep_Face_Recognition_CVPR_2021_paper.pdf
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        batch_size: int,
        s: float = 64.0,
        margin: float = 0.5,
        delta_t: int = 100,
        lambda_membank: float = 0.5,
        mem_bank_start_epoch: int = 2,
        accelerator: str = "cpu",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(VariationalPrototypeLearning, self).__init__(*args, **kwargs)
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.delta_t = delta_t
        self.lambda_membank = lambda_membank
        self.mem_bank_start_epoch = mem_bank_start_epoch
        if accelerator == "cuda":
            self.prototypes = torch.nn.Parameter(torch.cuda.FloatTensor(num_classes, embedding_size))  # type: ignore
        else:
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

    def set_using_memory_bank(self, using_memory_bank: bool) -> bool:
        """Sets whether or not to use the memory bank"""
        self.using_memory_bank = using_memory_bank
        return self.using_memory_bank

    def set_weights(self, weights: torch.Tensor) -> None:
        """Sets the weights of the prototypes"""
        assert weights.shape == self.prototypes.shape

        if torch.cuda.is_available() and self.prototypes.device != weights.device:
            weights = weights.cuda()

        self.prototypes = torch.nn.Parameter(weights)

    def update_memory_bank(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Updates the memory bank with the current batch of embeddings and labels"""

        if embeddings.shape[0] < self.batch_size:
            embeddings = torch.cat(
                (
                    embeddings,
                    torch.zeros(self.batch_size - embeddings.shape[0], self.embedding_size, device=embeddings.device),
                ),
                dim=0,
            )
            labels = torch.cat((labels, torch.zeros(self.batch_size - labels.shape[0], device=embeddings.device) - 1), dim=0)  # type: ignore

        if self.memory_bank.device != embeddings.device or self.memory_bank_labels.device != embeddings.device:
            self.memory_bank = self.memory_bank.to(embeddings.device)
            self.memory_bank_labels = self.memory_bank_labels.to(embeddings.device)

        self.memory_bank[self.memory_bank_ptr * self.batch_size : (self.memory_bank_ptr + 1) * self.batch_size] = (
            embeddings
        )
        self.memory_bank_labels[
            self.memory_bank_ptr * self.batch_size : (self.memory_bank_ptr + 1) * self.batch_size
        ] = labels
        self.memory_bank_ptr = (self.memory_bank_ptr + 1) % self.delta_t

    @torch.no_grad()
    def get_memory_bank_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the prototypes and their frequency in the memory bank"""
        prototypes = torch.zeros(self.num_classes, self.embedding_size, device=self.memory_bank.device)
        frequency = torch.zeros(self.num_classes, device=self.memory_bank.device)
        for i in range(self.num_classes):
            prototypes[i] = torch.mean(self.memory_bank[self.memory_bank_labels == i], dim=0)
            frequency[i] = torch.sum(self.memory_bank_labels == i)  # type: ignore

        # set to zero if frequency is zero
        prototypes[frequency == 0] = 0.0

        return prototypes, frequency

    def calculate_prototype(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates the prototype for the given embeddings and labels"""
        if self.using_memory_bank:
            mem_bank_prototypes, prototype_frequency = self.get_memory_bank_prototypes()
            mem_bank_prototypes = mem_bank_prototypes.to(embeddings.device)
            relative_frequency = prototype_frequency / (
                torch.sum(prototype_frequency) if torch.sum(prototype_frequency) > eps else 1.0
            )
            relative_frequency = relative_frequency.unsqueeze(1).repeat(1, self.embedding_size).to(embeddings.device)
            is_known = (relative_frequency > eps).float().to(embeddings.device)

            prototypes = (
                (1 - self.lambda_membank * is_known) * self.prototypes
            ) + self.lambda_membank * is_known * mem_bank_prototypes

            self.update_memory_bank(embeddings, labels)
        else:
            prototypes = self.prototypes

        if prototypes.device != embeddings.device:
            prototypes = prototypes.to(embeddings.device)

        return prototypes

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> gtypes.LossPosNegDist:
        """Forward pass of the Variational Prototype Learning loss function"""
        prototypes = self.calculate_prototype(embeddings, labels)

        cos_theta = (
            torch.nn.functional.normalize(embeddings).unsqueeze(1)
            * torch.nn.functional.normalize(prototypes).unsqueeze(0)
        ).sum(dim=2)

        sine_theta = torch.sqrt(
            torch.maximum(1.0 - torch.pow(cos_theta, 2), torch.tensor([eps], device=cos_theta.device))
        ).clamp(eps, 1.0 - eps)
        phi = (
            cos_theta * self.cos_m - sine_theta * self.sin_m
        )  # additionstheorem cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

        mask = torch.zeros(cos_theta.size(), device=cos_theta.device)
        mask.scatter_(1, labels.view(-1, 1).long(), 1)  # mask is one-hot encoded labels

        output = (mask * phi) + ((1.0 - mask) * cos_theta)  # NOTE: sometimes there is an additional penalty term
        output *= self.s
        loss = self.ce(output, labels)

        return loss, torch.Tensor([-1.0]), torch.Tensor([-1.0])  # dummy values for pos/neg distances
