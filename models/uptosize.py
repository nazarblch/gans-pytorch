import math
from typing import List

from torch import nn, Tensor
import torch
from torch.nn import functional as F
from gans_pytorch.stylegan2.model import Blur, ConvLayer, EqualLinear
from gans_pytorch.stylegan2.op import FusedLeakyReLU
from models.common import View
from models.stylegan import ScaledConvTranspose2d


class Uptosize(nn.Module):

    def __init__(self, channel1, channel2, size2):
        super().__init__()
        modules = [
            EqualLinear(channel1, channel1 * 4 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            View(channel1 // 2, 4, 4)
        ]
        tmp_size = 4
        tmp_channel = channel1//2
        min_nc = 16
        while tmp_size < size2:
            nc_next = max(min_nc, tmp_channel//2)
            modules.append(ScaledConvTranspose2d(tmp_channel, nc_next, 3))
            tmp_channel = nc_next
            tmp_size *= 2
        assert(tmp_size == size2)
        nc_next = max(min_nc, tmp_channel // 2)
        modules.append(ConvLayer(nc_next, channel2, 1))
        self.main = nn.Sequential(*modules)

    def forward(self, input: Tensor):
        return self.main(input.view(input.shape[0], input.shape[1]))


class UpsampleList(nn.Module):

    def __init__(self, num_layers: int, nc: int, nc_min: int):
        super().__init__()

        self.nc = nc

        self.upsamples = nn.ModuleList()
        self.upsamples.append(nn.Sequential(
            EqualLinear(nc, nc * 4 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            View(nc // 2, 4, 4)
        ))

        tmp_channel = nc // 2

        for i in range(num_layers-1):
            nc_next = max(nc_min, tmp_channel // 2)
            self.upsamples.append(
                ScaledConvTranspose2d(tmp_channel, nc_next, 3)
            )
            tmp_channel = nc_next

    def forward(self, vector: Tensor) -> List[Tensor]:
        batch = vector.shape[0]
        res = [
            self.upsamples[0](vector)
        ]

        for layer in self.upsamples[1:]:
            res.append(
                layer(res[-1])
            )

        return res


class MakeNoise(nn.Module):

    def __init__(self, num_layers: int, nc: int, noise_nc: List[int]):
        super().__init__()

        nc_min = 16
        self.upsample = UpsampleList(num_layers, nc, nc_min)
        self.to_noise = nn.ModuleList()
        tmp_channel = nc

        for i in range(num_layers):
            tmp_channel = max(nc_min, tmp_channel // 2)
            self.to_noise.append(
                ConvLayer(tmp_channel, noise_nc[i], 3)
            )

    def forward(self, vector: Tensor):

        noises = []

        for i, usample in enumerate(self.upsample(vector)):
            ni = self.to_noise[i](usample)
            noises.append(ni)
            if i > 0:
                noises.append(ni)

        return noises
