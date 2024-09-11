from typing import Literal, Union

import torch
import torch.fft
import torch.utils
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: Union[float, torch.Tensor, nn.Parameter] = 3.0, eps: float = 1e-6
    ) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).flatten(1)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GeM_adapted(nn.Module):
    def __init__(
        self, p: float = 3.0, p_shape: Union[tuple[int], int] = 1, eps: float = 1e-6
    ) -> None:  # TODO(rob2u): make p_shape variable (only 1 supported currently)
        super().__init__()
        self.p = nn.Parameter(torch.ones(p_shape) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: Union[float, torch.Tensor, nn.Parameter] = 3.0, eps: float = 1e-6
    ) -> torch.Tensor:  # TODO(rob2u): find better way instead of multiplying by sign
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).view(x.size(0), -1)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GAP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class FormatWrapper(nn.Module):
    def __init__(self, pool: nn.Module, format: Literal["NCHW", "NHWC"] = "NCHW") -> None:
        super().__init__()
        assert format in ("NCHW", "NHWC")

        self.pool = pool
        self.format = format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.format == "NHWC":
            x = x.permute(0, 2, 3, 1)
        elif self.format == "NCHW":
            x = x
        else:
            raise ValueError(f"Unknown format {self.format}")
        return self.pool(x)
