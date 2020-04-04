from typing import Callable, List, Tuple, Any, Union
import torch
from torch import nn, Tensor


class LambdaModule(nn.Module):
    def __init__(self,  module: List[nn.Module], f: Callable[[Tuple[Tensor, ...]], Tuple[Tensor, ...]]):
        super().__init__()
        self.module = nn.ModuleList(module)
        self.forward = f
