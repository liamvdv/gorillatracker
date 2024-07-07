from typing import Union

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
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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
        x_pow_mean = x.pow(p).mean((-2, -1))
        return x_pow_mean.abs().pow(1.0 / p).view(-1) * (x_pow_mean.view(-1).sign()) ** p

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


class FourierConvolution(nn.Module):
    def __init__(self, param_shape: tuple[int]) -> None:
        super().__init__()
        self.param_matrix = nn.Parameter(torch.randn(param_shape) + 1j * torch.randn(param_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft2(x)
        x_fft_filtered = x_fft * self.param_matrix
        x_ifft = torch.fft.irfft2(x_fft_filtered)
        return x_ifft
