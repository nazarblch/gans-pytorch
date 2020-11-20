import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from op import FusedLeakyReLU
from stylegan2.model import EqualLinear, EqualConv2d, NoiseInjection, StyledConv, Blur


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

        factor = 2
        blur_kernel = [1, 3, 3, 1]
        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1

        self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input: Tensor):

        weight = self.scale * self.weight
        out = F.conv_transpose2d(input, weight, padding=0, stride=2)

        return self.activate(self.blur(out))

