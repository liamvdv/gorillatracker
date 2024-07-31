from typing import Any, Dict, List, Literal, Optional, Union

import torch

import gorillatracker.type_helper as gtypes
from gorillatracker.utils.labelencoder import LinearSequenceEncoder

eps = 1e-8  # an arbitrary small value to be used for numerical stability


class FocalLoss(torch.nn.Module):
    def __init__(
        self, num_classes: int = 182, gamma: float = 2.0, label_smoothing: float = 0.0, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()
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
        purpose: Literal["val", "train"] = "train",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.s = s
        self.angle_margin = torch.tensor([angle_margin]).to(accelerator)
        self.additive_margin = torch.tensor([additive_margin]).to(accelerator)
        self.cos_m = torch.cos(torch.tensor([angle_margin])).to(accelerator)
        self.sin_m = torch.sin(torch.tensor([angle_margin])).to(accelerator)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.k_subcenters = k_subcenters
        self.class_distribution = class_distribution
        self.use_class_weights = use_class_weights
        self.num_samples = (
            (sum([class_distribution[label] for label in class_distribution.keys()]) if self.class_distribution else 0)
            if self.use_class_weights
            else 1
        )
        self.purpose = purpose
        self.accelerator = accelerator

        self.prototypes: Union[torch.nn.Parameter, torch.Tensor]
        if self.purpose == "train":
            self.prototypes = torch.nn.Parameter(
                torch.zeros(
                    (k_subcenters, num_classes, embedding_size),
                    device=accelerator,
                    dtype=torch.float32,
                ),
                requires_grad=True,
            )
            tmp_rng = torch.Generator(device=accelerator)
            torch.nn.init.xavier_uniform_(self.prototypes, generator=tmp_rng)
        else:
            self.prototypes = torch.zeros(
                (k_subcenters, num_classes, embedding_size), device=accelerator, dtype=torch.float32
            )

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
        labels_onehot: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> gtypes.LossPosNegDist:
        """Forward pass of the ArcFace loss function"""
        embeddings = embeddings.to(self.accelerator)
        assert self.prototypes.device == embeddings.device, "Prototypes and embeddings must be on the same device"
        assert not any(torch.flatten(torch.isnan(embeddings))), "NaNs in embeddings"

        # NOTE(rob2u): necessary for range 0:n-1
        # get class frequencies
        class_freqs = torch.ones_like(labels, device=embeddings.device)
        if self.use_class_weights and self.purpose == "train":
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
        cos_theta = cos_theta.to(embeddings.device)
        
        sine_theta = torch.sqrt(
            torch.maximum(
                1.0 - torch.pow(cos_theta, 2),
                torch.tensor([eps], device=cos_theta.device),
            )
        ).clamp(eps, 1.0 - eps)
        
        if self.cos_m.device != embeddings.device: # HACK
            self.cos_m = self.cos_m.to(embeddings.device)
            self.sin_m = self.sin_m.to(embeddings.device)
            self.additive_margin = self.additive_margin.to(embeddings.device)

        phi = (
            self.cos_m.unsqueeze(1).unsqueeze(2) * cos_theta - self.sin_m.unsqueeze(1).unsqueeze(2) * sine_theta
        )  # additionstheorem cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        phi = phi - self.additive_margin.unsqueeze(0)

        mask = torch.zeros(
            (cos_theta.shape[0], self.num_classes, self.k_subcenters), device=cos_theta.device
        )  # batch x num_classes x k_subcenters
        mask.scatter_(1, labels.view(1, -1, 1).long(), 1)

        output = (mask * phi) + ((1.0 - mask) * cos_theta)  # NOTE: the margin is only added to the correct class
        output *= self.s
        output = torch.mean(output, dim=2)  # batch x num_classes

        assert not any(torch.flatten(torch.isnan(output))), "NaNs in output"
        loss = self.ce(output, labels) if labels_onehot is None else self.ce(output, labels_onehot)
        
        loss = loss * (1 / class_freqs)  # NOTE: class_freqs is a tensor of class frequencies
        loss = torch.mean(loss)

        assert not any(torch.flatten(torch.isnan(loss))), "NaNs in loss"
        return loss, torch.Tensor([-1.0]), torch.Tensor([-1.0])  # dummy values for pos/neg distances

    def update(self, weights: torch.Tensor, num_classes: int, le: LinearSequenceEncoder) -> None:
        """Sets the weights of the prototypes"""

        assert self.purpose == "val", "Manually setting the prototypes is only allowed for validation"

        self.num_classes = num_classes
        self.le = le

        weights = weights.unsqueeze(0)

        if torch.cuda.is_available() and self.prototypes.device != weights.device:
            weights = weights.cuda()

        self.prototypes = weights


class ElasticArcFaceLoss(ArcFaceLoss):
    def __init__(
        self,
        margin_sigma: float = 0.01,
        accelerator: Literal["cuda", "cpu", "tpu", "mps"] = "cpu",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(accelerator=accelerator, *args, **kwargs)  # type: ignore
        self.margin_sigma = torch.tensor([margin_sigma]).to(accelerator)
        self.is_eval = False
        self.accelerator = accelerator

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        labels_onehot: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> gtypes.LossPosNegDist:
        angle_margin = torch.tensor([self.angle_margin], device=embeddings.device)
        self.margin_sigma = torch.tensor([self.margin_sigma], device=embeddings.device)

        if not self.is_eval:
            angle_margin = (
                angle_margin + torch.randn_like(labels, dtype=torch.float32, device=embeddings.device) * self.margin_sigma
            ) # batch -> scale by self.margin_sigma

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
        super().__init__(*args, **kwargs)
        self.is_eval = False
        self.h = h
        self.m1 = self.angle_margin
        self.m2 = self.additive_margin
        self.norm = torch.nn.BatchNorm1d(1, affine=False, momentum=momentum).to(kwargs.get("accelerator", "cpu"))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        labels_onehot: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> gtypes.LossPosNegDist:
        if self.norm.running_mean.device != embeddings.device:  # type: ignore
            self.norm = self.norm.to(embeddings.device)
            
        if self.m1.device != embeddings.device:
            self.m1 = self.m1.to(embeddings.device)
            self.m2 = self.m2.to(embeddings.device)

        if not self.is_eval:
            g = (embeddings.detach() ** 2).sum(dim=1).sqrt()
            g = self.norm(g.unsqueeze(1)).squeeze(1)
            g = torch.clamp(g / self.h, -1, 1)
            g = g.to(embeddings.device)
            g_angle = -self.m1 * g
            g_additive = self.m2 * g + self.m2

            self.cos_m = torch.cos(g_angle)
            self.sin_m = torch.sin(g_angle)
            self.additive_margin = g_additive
        return super().forward(embeddings, labels, labels_onehot=labels_onehot, **kwargs)

    def eval(self) -> Any:
        self.is_eval = True
        return super().eval()
