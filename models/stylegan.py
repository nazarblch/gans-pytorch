import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from op import FusedLeakyReLU
from stylegan2.model import EqualLinear, EqualConv2d, NoiseInjection, StyledConv, Blur


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


class ModulatedResBlock(nn.Module):
    def __init__(self, dim, style_dim):
        super(ModulatedResBlock, self).__init__()

        self.conv1 = StyledConv(dim, dim, 3, style_dim) #ModulatedConv2d(dim, dim, 3, style_dim)
        self.conv2 = StyledConv(dim, dim, 3, style_dim) #ModulatedConv2d(dim, dim, 3, style_dim)
        self.res = EqualConv2d(dim, dim, 1, 1, 0)
        self.noise1 = nn.Parameter(torch.randn(1, 1, 64, 64), requires_grad=False)
        self.noise2 = nn.Parameter(torch.randn(1, 1, 64, 64), requires_grad=False)


    def forward(self, x: Tensor, style: Tensor):
        residual = x
        # out = self.activation(
        #         self.module_noise(self.conv1(x, style), self.noise1)
        # )
        # out = self.activation(
        #     self.module_noise(
        #         self.conv2(out, style),
        #         self.noise2
        #     )
        # )
        # out = self.conv1(x, style)
        out = self.conv1(x, style, self.noise1)
        # out = self.conv2(out, style)
        out = self.conv2(out, style, self.noise2)
        out = (out + self.res(residual)) / math.sqrt(2)
        return out


class ModulatedResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, style_dim, norm=None):
        super(ModulatedResBlocks, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_blocks):
            self.model.append(ModulatedResBlock(dim, style_dim))


    def forward(self, x, style):
        for i in range(len(self.model)):
            x = self.model[i](x, style)
        return x
