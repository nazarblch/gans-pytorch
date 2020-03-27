import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from stylegan2.model import EqualLinear


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size),
            requires_grad=True
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}'
        )

    def forward(self, content: Tensor, style: Tensor):
        batch, in_channel, height, width = content.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        content = content.view(1, batch * in_channel, height, width)
        out = F.conv2d(content, weight, padding=self.padding, groups=batch)

        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


class ScaledConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size
    ):

        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size),
            requires_grad=True
        )

    def forward(self, input: Tensor):

        weight = self.scale * self.weight
        out = F.conv_transpose2d(input, weight, padding=0, stride=2)

        return out
