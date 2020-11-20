import torch
from torch import nn, Tensor

from model import EqualConv2d, EqualLinear
from nn.common.view import View


class StyleEncoder(nn.Module):
    def __init__(self, style_dim):
        super(StyleEncoder, self).__init__()
        self.model = [
            EqualConv2d(3, 16, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1),
            EqualLinear(256 * 4 * 4, style_dim * 2, activation="fused_lrelu"),
            EqualLinear(style_dim * 2, style_dim * 2),
            View(2, style_dim)
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)