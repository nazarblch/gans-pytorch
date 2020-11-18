from typing import List, Dict, Callable, Tuple, Optional, TypeVar, Generic, Any
import math
import torch
from torch import Tensor, nn
from gan.discriminator import Discriminator
from models.common import View
from models.stylegan import ScaledConvTranspose2d
from models.uptosize import MakeNoise, Uptosize
from stylegan2.model import Generator as StyleGenerator2, EqualLinear, ConvLayer, EqualConv2d, StyledConv, ResBlock
from stylegan2.model import Discriminator as StyleDiscriminator2
from gan.generator import Generator


class CondGen2(nn.Module):

    def __init__(self, gen: Generator):
        super().__init__()

        self.gen: Generator = gen

        self.noise = MakeNoise(7, 140, [512, 512, 512, 512, 256, 128, 64])

        self.condition_preproc = nn.Sequential(
            EqualLinear(140, 256 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, 4, 4),
            EqualConv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.inject_index = 2

    def decode(self, cond: Tensor, latent: Tensor):
        noise = self.noise(cond)
        input = self.condition_preproc(cond)
        latent = [latent[:, 0], latent[:, 1]]
        return self.gen(latent, condition=input, noise=noise, input_is_latent=True, inject_index=self.inject_index)[0]

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):
        noise = self.noise(cond)
        input = self.condition_preproc(cond)

        # for i in range(len(noise)):
        #     if i > 1 and i % 2 == 0:
        #         noise[i] = None

        return self.gen(z, condition=input, noise=noise, return_latents=return_latents, inject_index=self.inject_index)


class CondGen3(nn.Module):

    def __init__(self, gen: StyleGenerator2, heatmap_channels: int, cond_mult: float = 10):
        super().__init__()

        self.cond_mult = cond_mult

        self.gen: Generator = gen
        self.noise_up = nn.ModuleList([
            ScaledConvTranspose2d(heatmap_channels, self.gen.channels[128], 3),
            ScaledConvTranspose2d(self.gen.channels[128], self.gen.channels[256], 3),
        ])
        self.noise_down = nn.ModuleList([
            ConvLayer(heatmap_channels, self.gen.channels[64], 3, downsample=False),
            ConvLayer(self.gen.channels[64], self.gen.channels[32], 3, downsample=True),
            ConvLayer(self.gen.channels[32], self.gen.channels[16], 3, downsample=True),
            ConvLayer(self.gen.channels[16], self.gen.channels[8], 3, downsample=True),
            ConvLayer(self.gen.channels[8], self.gen.channels[4], 3, downsample=True)
        ])
        self.inject_index = 2

    def make_noise(self, heatmap: Tensor):
        x = heatmap * self.cond_mult
        noise_up_list = []
        for i in self.noise_up:
            x = i.forward(x)
            noise_up_list.append(x)
            noise_up_list.append(x)

        y = heatmap * self.cond_mult
        noise_down_list = []
        for i in self.noise_down:
            y = i.forward(y)
            noise_down_list.append(y)
            noise_down_list.append(y)

        return noise_down_list[-2::-1] + noise_up_list

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):
        noise = self.make_noise(cond)
        return self.gen(z, condition=noise[0], noise=noise, return_latents=return_latents,
                        inject_index=self.inject_index)


class CondGen7(nn.Module):

    def __init__(self, gen: StyleGenerator2, heatmap_channels: int, cond_mult: float = 10):
        super().__init__()

        self.cond_mult = cond_mult

        self.gen: Generator = gen

        self.init_cov = ConvLayer(heatmap_channels, self.gen.channels[256], kernel_size=1)

        self.noise = nn.ModuleList([
            ConvLayer(self.gen.channels[256], self.gen.channels[256], 3, downsample=False),
            ConvLayer(self.gen.channels[256], self.gen.channels[128], 3, downsample=True),
            ConvLayer(self.gen.channels[128], self.gen.channels[64], 3, downsample=True),
            ConvLayer(self.gen.channels[64], self.gen.channels[32], 3, downsample=True),
            ConvLayer(self.gen.channels[32], self.gen.channels[16], 3, downsample=True),
            ConvLayer(self.gen.channels[16], self.gen.channels[8], 3, downsample=True),
            ConvLayer(self.gen.channels[8], self.gen.channels[4], 3, downsample=True)
        ])
        self.inject_index = 2

    def make_noise(self, heatmap: Tensor):
        x = self.init_cov(heatmap)

        noise_down_list = []
        for i in self.noise:
            x = i.forward(x)
            noise_down_list.append(x)
            noise_down_list.append(x)

        return noise_down_list[-2::-1]

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):
        noise = self.make_noise(cond)
        return self.gen(z, condition=noise[0], noise=noise, return_latents=return_latents,
                        inject_index=self.inject_index)


class CondGenDecode(nn.Module):
    def __init__(self, gen: CondGen3):
        super().__init__()
        self.gen: CondGen3 = gen

    def forward(self, cond: Tensor, latent: Tensor):
        noise = self.gen.make_noise(cond)
        latent = [latent[:, 0], latent[:, 1]]
        return self.gen.gen(latent, condition=noise[0], noise=noise, input_is_latent=True, inject_index=self.gen.inject_index)[0]

