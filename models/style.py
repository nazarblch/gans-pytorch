import torch
from torch import nn
import random


class StyleNoise(nn.Module):

    def __init__(self, in_dim=512, out_dim=512, n_mlp=8):
        super().__init__()

        layers = [PixelNorm(), EqualLinear(in_dim, out_dim), nn.LeakyReLU(0.2)]
        for i in range(1, n_mlp):
            layers.append(EqualLinear(out_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, step, mean_style=None, style_weight=0, mixing_range=(-1, -1)):

        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        styles_stack = []

        if len(styles) < 2:
            inject_index = [10]

        else:
            inject_index = random.sample(list(range(step)), len(styles) - 1)

        crossover = 0

        for i in range(step + 1):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(styles))

                style_step = styles[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = styles[1]

                else:
                    style_step = styles[0]

            styles_stack.append(style_step)

        return styles_stack


class Noise:

    def forward(self, input, step, noise=None):

        if type(input) not in (list, tuple):
            input = [input]

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        return noise


class GeneratorWithStyle(nn.Module):
    def __init__(self, alpha, step):
        super().__init__()
        self.generator = GeneratorBuilder(alpha=alpha).build(step)
        self.style: StyleNoise = StyleNoise()
        self.noise = Noise()
        self.step = step

    def forward(self, input, noise=None, mean_style=None, style_weight=0, mixing_range=(-1, -1)):

        noise = self.noise.forward(input, self.step, noise)
        style = self.style.forward(input, self.step, mean_style, style_weight, mixing_range)

        gen_input = {"input": noise[0]}

        for i in range(self.step+1):
            gen_input[f"style{i+1}"] = style[i]
            gen_input[f"noise{i+1}"] = noise[i]

        return self.generator(gen_input)[f"out_rgb{self.step+1}"]

