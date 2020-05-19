from abc import ABC, abstractmethod
from typing import List, Callable, TypeVar, Generic, Optional, Type
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.unet.la_divina_progressiya import Progressive, ElementwiseModuleList, ReverseListCollector, \
    LastElementCollector


class UNet2(nn.Sequential):

    def __init__(self,
                 down_blocks: List[nn.Module],
                 up_blocks: List[nn.Module]):
        super().__init__(
            Progressive[List[Tensor]](down_blocks, ReverseListCollector),
            Progressive[List[Tensor]](up_blocks),
        )


class UNet4(nn.Sequential):

    def __init__(self,
                 down_blocks: List[nn.Module],
                 middle_block: List[nn.Module],
                 up_blocks: List[nn.Module],
                 final_blocks: List[nn.Module]):
        super().__init__(
            Progressive[List[Tensor]](down_blocks, ReverseListCollector),
            ElementwiseModuleList[List[Tensor]](middle_block),
            Progressive[List[Tensor]](up_blocks),
            Progressive[Tensor](final_blocks, LastElementCollector)
        )






