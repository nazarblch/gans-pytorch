import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from math import sqrt
import random
from style_gan.models.other import StyledConvBlock, EqualConv2d
from style_gan.models.builder import ModuleBuilder, Identity


class AlphaMix(nn.Module):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, t1: Tensor, t2: Tensor):
        assert t1.shape[-1] * 2 == t2.shape[-1]
        t1 = F.interpolate(t1, scale_factor=2, mode='nearest')
        return (1 - self.alpha) * t1 + self.alpha * t2


class GeneratorBuilder:

    def __init__(self, fused=True, alpha: float = -1):

        builder3 = ModuleBuilder()

        builder3.add_module("input_1", Identity(), ["input", "style1", "noise1"], ["out1", "style1", "noise1"])
        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("input_%d" % i, Identity(), ["style%d" % i, "noise%d" % i], ["style%d" % i, "noise%d" % i])
        )

        builder3.add_modules(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("progression_%d" % i,
                       ["out%d" % i, "style%d" % i, "noise%d" % i],
                       ["out%d" % (i + 1)])
        )(
            StyledConvBlock(128, 128, 3, 1, initial=True),   # 4
            StyledConvBlock(128, 128, 3, 1, upsample=True),  # 8
            StyledConvBlock(128, 128, 3, 1, upsample=True),  # 16
            StyledConvBlock(128, 64, 3, 1, upsample=True),  # 32
            StyledConvBlock(64, 64, 3, 1, upsample=True),  # 64
            StyledConvBlock(64, 64, 3, 1, upsample=True, fused=fused),  # 128
            StyledConvBlock(64, 64, 3, 1, upsample=True, fused=fused),  # 256
            StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
            StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
        )

        builder3.add_edge(["input_1"], "progression_1")
        builder3.add_edge_seq([2, 3, 4, 5, 6, 7, 8, 9],
                              lambda i: ([f"input_{i}", f"progression_{i - 1}"], f"progression_{i}"))

        builder3.add_modules(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("to_rgb_%d" % i,
                       ["out%d" % (i + 1)],
                       ["out_rgb%d" % i])
        )(
            EqualConv2d(128, 3, 1),
            EqualConv2d(128, 3, 1),
            EqualConv2d(128, 3, 1),
            EqualConv2d(64, 3, 1),
            EqualConv2d(64, 3, 1),
            EqualConv2d(64, 3, 1),
            EqualConv2d(64, 3, 1),
            EqualConv2d(32, 3, 1),
            EqualConv2d(16, 3, 1),
        )

        builder3.add_edge_seq([1, 2, 3, 4, 5, 6, 7, 8, 9],
                              lambda i: ([f"progression_{i}"], f"to_rgb_{i}"))

        builder3.add_module("alpha_mix_1", Identity(), ["out_rgb1"], ["out_rgb1"])
        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("alpha_mix_%d" % i,
                       AlphaMix(alpha),
                       ["out_rgb%d" % (i - 1), "out_rgb%d" % i],
                       ["out_rgb%d" % i])
        )

        builder3.add_edge_seq([2, 3, 4, 5, 6, 7, 8, 9], lambda i: ([f"to_rgb_{i - 1}", f"to_rgb_{i}"], f"alpha_mix_{i}"))

        self.builder = builder3
        self.builder.cuda()

    def build(self, step):

        return self.builder.build(
            ["input_" + str(i+1) for i in range(step + 1)],
            "alpha_mix_" + str(step + 1)
        )
