from typing import Callable, List, Tuple

import torch
from torch import nn


class LambdaF(nn.Module):
    def __init__(self,  module: List[nn.Module], f: Callable[[Tuple[torch.Tensor]], torch.Tensor]):
        super().__init__()
        self.module = nn.ModuleList(module)
        self.f = f

    def forward(self, *input: torch.Tensor):
        return self.f(*input)
