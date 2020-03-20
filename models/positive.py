import torch
from torch import nn, Tensor
from torch.nn import functional as F


class PosLinear(nn.Linear):

    def forward(self, input: Tensor):
        return F.linear(input, torch.abs(self.weight), self.bias)


class PosConv2d(nn.Conv2d):

    def forward(self, input: Tensor):
        return self.conv2d_forward(input, torch.abs(self.weight))
