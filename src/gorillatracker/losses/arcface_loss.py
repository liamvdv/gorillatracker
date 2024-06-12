import math
from typing import Any, Dict, List, Literal, Tuple, Union

import torch

import gorillatracker.type_helper as gtypes
from gorillatracker.utils.labelencoder import LinearSequenceEncoder

eps = 1e-8  # an arbitrary small value to be used for numerical stability


class FocalLoss(torch.nn.Module):
    def __init__(
        self, num_classes: int = 182, gamma: float = 2.0, label_smoothing: float = 0.0, *args: Any, **kwargs: Any
    ) -> None:
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # assert len(alphas) == len(target), "Alphas must be the same length as the target"
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class ArcFaceLoss(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(
        self,
        embedding_size: int,
        num_classes: int = 182,
        class_distribution: Union[Dict[int, int]] = {},
        s: float = 64.0,
        angle_margin: float = 0.5,
        additive_margin: float = 0.0,
        accelerator: Literal["cuda", "cpu", "tpu", "mps"] = "cpu",
        k_subcenters: int = 2,
        use_focal_loss: bool = False,
        label_smoothing: float = 0.0,
        use_class_weights: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.angle_margin = torch.Tensor([angle_margin]).to(accelerator)
        self.additive_margin = torch.Tensor([additive_margin]).to(accelerator)
        self.cos_m = torch.cos(torch.Tensor([angle_margin])).to(accelerator)
        self.sin_m = torch.sin(torch.Tensor([angle_margin])).to(accelerator)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.k_subcenters = k_subcenters
        self.class_distribution = class_distribution
        self.use_class_weights = use_class_weights
        self.num_samples = (
            sum([class_distribution[label] for label in class_distribution.keys()]) if self.class_distribution else 0
        )

        self.accelerator = accelerator
        self.prototypes = torch.nn.Parameter(
            torch.zeros(
                (k_subcenters, num_classes, embedding_size),
                device=accelerator,
                dtype=torch.float32,
            )
        )

        tmp_rng = torch.Generator(device=accelerator)
        torch.nn.init.xavier_uniform_(self.prototypes, generator=tmp_rng)
        self.ce: Union[FocalLoss, torch.nn.CrossEntropyLoss]
        if use_focal_loss:
            self.ce = FocalLoss(num_classes=num_classes, label_smoothing=label_smoothing, *args, **kwargs)  # type: ignore
        else:
            self.ce = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

        self.le = LinearSequenceEncoder()  # NOTE: new instance (range 0:num_classes-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        labels_onehot: torch.Tensor = torch.Tensor(),
        **kwargs: Any,
    ) -> gtypes.LossPosNegDist:
        """Forward pass of the ArcFace loss function"""
        embeddings = embeddings.to(self.accelerator)
        assert self.prototypes.device == embeddings.device, "Prototypes and embeddings must be on the same device"
        assert not any(torch.flatten(torch.isnan(embeddings))), "NaNs in embeddings"

        # NOTE(rob2u): necessary for range 0:n-1
        # get class frequencies
        class_freqs = torch.ones_like(labels)
        if self.use_class_weights:
            class_freqs = torch.tensor([self.class_distribution[label.item()] for label in labels]).to(
                embeddings.device
            )
            class_freqs = class_freqs.float() / self.num_samples
            class_freqs = class_freqs.clamp(eps, 1.0)

        labels_transformed: List[int] = self.le.encode_list(labels.tolist())
        labels = torch.tensor(labels_transformed).to(embeddings.device)

        cos_theta = torch.einsum(
            "bj,knj->bnk",
            torch.nn.functional.normalize(embeddings, dim=-1),
            torch.nn.functional.normalize(self.prototypes, dim=-1),
        )  # batch x num_classes x k_subcenters

        sine_theta = torch.sqrt(
            torch.maximum(
                1.0 - torch.pow(cos_theta, 2),
                torch.tensor([eps], device=cos_theta.device),
            )
        ).clamp(eps, 1.0 - eps)
        phi = (
            self.cos_m.unsqueeze(1) * cos_theta - self.sin_m.unsqueeze(1) * sine_theta
        )  # additionstheorem cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        phi = phi - self.additive_margin.unsqueeze(1)

        mask = torch.zeros(
            (cos_theta.shape[0], self.num_classes, self.k_subcenters), device=cos_theta.device
        )  # batch x num_classes x k_subcenters
        mask.scatter_(1, labels.view(1, -1, 1).long(), 1)

        output = (mask * phi) + ((1.0 - mask) * cos_theta)  # NOTE: the margin is only added to the correct class
        output *= self.s
        output = torch.mean(output, dim=2)  # batch x num_classes

        assert not any(torch.flatten(torch.isnan(output))), "NaNs in output"
        loss = self.ce(output, labels) if len(labels_onehot) == 0 else self.ce(output, labels_onehot)
        if self.use_class_weights:
            loss = loss * (1 / class_freqs)  # NOTE: class_freqs is a tensor of class frequencies
        loss = torch.mean(loss)

        assert not any(torch.flatten(torch.isnan(loss))), "NaNs in loss"
        return loss, torch.Tensor([-1.0]), torch.Tensor([-1.0])  # dummy values for pos/neg distances

    def update(self, weights: torch.Tensor, num_classes: int, le: LinearSequenceEncoder) -> None:
        """Sets the weights of the prototypes"""
        self.num_classes = num_classes
        self.le = le

        weights = weights.unsqueeze(0)

        if torch.cuda.is_available() and self.prototypes.device != weights.device:
            weights = weights.cuda()

        self.prototypes = torch.nn.Parameter(weights)


class ElasticArcFaceLoss(ArcFaceLoss):
    def __init__(self, margin_sigma: float = 0.01, *args: Any, **kwargs: Any) -> None:
        super(ElasticArcFaceLoss, self).__init__(*args, **kwargs)
        self.margin_sigma = margin_sigma
        self.is_eval = False

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        labels_onehot: torch.Tensor = torch.Tensor(),
        **kwargs: Any,
    ) -> gtypes.LossPosNegDist:
        angle_margin = torch.Tensor([self.angle_margin]).to(embeddings.device)
        if not self.is_eval:
            angle_margin = (self.angle_margin + torch.randn_like(labels, dtype=torch.float32) * self.margin_sigma).to(
                embeddings.device
            )  # batch -> scale by self.margin_sigma

        self.cos_m = torch.cos(angle_margin)
        self.sin_m = torch.sin(angle_margin)
        return super().forward(
            embeddings,
            labels,
            labels_onehot=labels_onehot,
        )

    def eval(self) -> Any:
        self.is_eval = True
        return super().eval()


class AdaFaceLoss(ArcFaceLoss):
    def __init__(self, momentum: float = 0.01, h: float = 0.33, *args: Any, **kwargs: Any) -> None:
        super(AdaFaceLoss, self).__init__(*args, **kwargs)
        self.is_eval = False
        self.h = h
        self.m1 = self.angle_margin
        self.m2 = self.additive_margin
        self.norm = torch.nn.BatchNorm1d(1, affine=False, momentum=momentum).to(kwargs.get("accelerator", "cpu"))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        labels_onehot: torch.Tensor = torch.Tensor(),
        **kwargs: Any,
    ) -> gtypes.LossPosNegDist:
        if self.norm.running_mean.device != embeddings.device:  # type: ignore
            self.norm = self.norm.to(embeddings.device)

        if not self.is_eval:
            g = (embeddings.detach() ** 2).sum(dim=1).sqrt()
            g = self.norm(g.unsqueeze(1)).squeeze(1)
            g = torch.clamp(g / self.h, -1, 1)
            g_angle = -self.m1 * g
            g_additive = self.m2 * g + self.m2

            self.cos_m = torch.cos(g_angle)
            self.sin_m = torch.sin(g_angle)
            self.additive_margin = g_additive
        return super().forward(embeddings, labels, labels_onehot=labels_onehot, **kwargs)

    def eval(self) -> Any:
        self.is_eval = True
        return super().eval()


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
            self.prototypes = torch.nn.Parameter(
                torch.zeros((num_classes, embedding_size), device="cuda", dtype=torch.float32)
            )
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
        self.le = LinearSequenceEncoder()  # NOTE: new instance (range 0:num_classes-1)

    def set_using_memory_bank(self, using_memory_bank: bool) -> bool:
        """Sets whether or not to use the memory bank"""
        self.using_memory_bank = using_memory_bank
        return self.using_memory_bank

    def set_weights(self, weights: torch.Tensor) -> None:
        """Sets the weights of the prototypes"""
        raise NotImplementedError("This method is not implemented for VariationalPrototypeLearning ask @rob2u for help")

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
            labels = torch.cat(
                (labels, torch.zeros(self.batch_size - labels.shape[0], device=embeddings.device) - 1), dim=0
            )

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
            frequency[i] = torch.sum(self.memory_bank_labels == i)

        # set to zero if frequency is zero
        prototypes[frequency == 0] = eps

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

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, images: torch.Tensor = torch.Tensor()
    ) -> gtypes.LossPosNegDist:
        """Forward pass of the Variational Prototype Learning loss function"""

        # NOTE(rob2u): necessary for range 0:n-1
        labels_transformed: List[int] = self.le.encode_list(labels.tolist())
        labels = torch.tensor(labels_transformed, device=embeddings.device)

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
