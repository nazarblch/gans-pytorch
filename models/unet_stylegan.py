from abc import ABC, abstractmethod
from typing import List, Callable, TypeVar, Generic, Optional, Type
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

TLT = TypeVar("TLT", Tensor, List[Tensor])


class TensorCollector(ABC, Generic[TLT]):
    @abstractmethod
    def append(self, t: Tensor) -> None:
        pass

    @abstractmethod
    def result(self) -> TLT:
        pass


class ListCollector(TensorCollector[List[Tensor]]):

    def __init__(self):
        self.data = []

    def result(self) -> List[Tensor]:
        out = self.data
        self.data = []
        return out

    def append(self, t: Tensor) -> None:
        self.data.append(t)


class ReverseListCollector(ListCollector):

    def result(self) -> List[Tensor]:
        self.data.reverse()
        out = self.data
        self.data = []
        return out


class LastElementCollector(TensorCollector[Tensor]):

    def __init__(self):
        self.data: Optional[Tensor] = None

    def result(self) -> Tensor:
        out = self.data
        self.data = None
        return out

    def append(self, t: Tensor) -> None:
        self.data = t


class RegressiveModuleList(nn.Module, Generic[TLT]):
    def __init__(self,
                 blocks: List[nn.Module],
                 collector_class: Type[TensorCollector[TLT]] = ListCollector):
        super(RegressiveModuleList, self).__init__()
        self.model_list = nn.ModuleList(blocks)
        self.collector_class = collector_class

    def forward(self, input: List[Tensor]) -> TLT:
        collector: TensorCollector[TLT] = self.collector_class()
        x = input[0]
        i = 0
        while i < (len(input) - 1):
            x = self.model_list[i](x)
            collector.append(x)
            x = torch.cat([x, input[i+1]], dim=1)
            i += 1
        while i < len(self.model_list):
            x = self.model_list[i](x)
            collector.append(x)
            i += 1
        return collector.result()


# class UNet(nn.Module):
#
#     def __init__(self, down_blocks: List[nn.Module], up_blocks: List[nn.Module]):
#         super().__init__()
#         self.down = RegressiveList(down_blocks)
#         self.up = RegressiveList(up_blocks)
#
#     def forward(self, x: List[Tensor]) -> List[Tensor]:
#         return self.up(self.down(x).reverse())
