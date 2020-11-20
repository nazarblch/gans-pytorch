import math
import random
from typing import List

import torch
from torch import Tensor, nn
from nn.common.view import View
from gan.nn.stylegan.components import ScaledConvTranspose2d
from gan.nn.stylegan.uptosize import UpsampleList
from nn.progressiya.base import Progressive, ProgressiveWithoutState, InjectByName, LastElementCollector
from nn.progressiya.unet import ProgressiveSequential, ZapomniKak, InputFilterName, InputFilterVertical, CopyKwToArgs
from stylegan2.model import Generator as StyleGenerator2, EqualLinear, ConvLayer, EqualConv2d, PixelNorm, ConstantInput, \
    StyledConv, ToRGB
from gan.generator import Generator


class NoiseToStyle(nn.Module):

    def __init__(self, style_dim, n_mlp, lr_mlp, n_latent):
        super().__init__()

        self.n_latent = n_latent

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self, z, inject_index):

        styles = [self.style(zi) for zi in z]

        if len(styles) < 2:
            inject_index = self.n_latent
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return [latent[:, i] for i in range(self.n_latent)]





class Generator1(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim

        self.channels = {
            4: 512,
            8: 512,
            16: 256,
            32: 256,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        convs = []
        # self.upsamples = nn.ModuleList()
        to_rgbs = []
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.progression = ProgressiveSequential(
            CopyKwToArgs({"z"}),
            InputFilterName({"inject_index"}),
            NoiseToStyle(style_dim, n_mlp, lr_mlp, self.n_latent),
            ZapomniKak("style"),
            CopyKwToArgs({"cond"}),
            InputFilterName({'noise', 'style'}),
            Progressive[List[Tensor]]([conv1] + convs, InjectByName("input")),
            ZapomniKak("input"),
            InputFilterName({'input', 'style'}),
            InputFilterVertical(list(range(1, len(convs) + 2, 2))),
            ProgressiveWithoutState[Tensor]([to_rgb1] + to_rgbs, InjectByName("skip"), LastElementCollector),
            return_keys=["style"]
        )

    def forward(
        self,
        z,
        condition,
        noise,
        inject_index=None
    ):

        image, latent = self.progression.forward(cond=condition, z=z, noise=noise, inject_index=inject_index)
        return image


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

