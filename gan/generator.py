from abc import ABC, abstractmethod
from typing import Optional, List
import torch
from torch import Tensor, nn


class Generator(nn.Module, ABC):

    def __init__(self):
        super(Generator, self).__init__()

    @abstractmethod
    def forward(self, *noise: Tensor) -> Tensor: pass





