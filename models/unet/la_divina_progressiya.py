from abc import ABC, abstractmethod
from typing import List, Callable, TypeVar, Generic, Optional, Type, Union, Tuple, Dict, Any, Set
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


class StateInjector(ABC):
    @abstractmethod
    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]): pass


class InjectByName(StateInjector):

    def __init__(self, name):
        self.name = name

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        kw[self.name] = state
        return args, kw


class InjectLast(StateInjector):

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        return (*args, state), kw


class InjectHead(StateInjector):

    def inject(self, state: Tensor, args: Tuple[Tensor, ...], kw: Dict[str, Tensor]):
        return (state, *args), kw


class Progressive(nn.Module, Generic[TLT]):
    def __init__(self,
                 blocks: List[nn.Module],
                 injector: StateInjector,
                 collector_class: Type[TensorCollector[TLT]] = ListCollector
                 ):
        super(Progressive, self).__init__()
        self.model_list = nn.ModuleList(blocks)
        self.collector_class = collector_class
        self.injector = injector

    def forward(self, state: Tensor, *args: List[Tensor], **kw: List[Tensor]) -> TLT:
        collector: TensorCollector[TLT] = self.collector_class()
        collector.append(state)
        i = 0
        while i < len(self.model_list):
            kw_i: Dict[str, Tensor] = dict((k, kw[k][i]) for k in kw.keys())
            args_i: Tuple[Tensor, ...] = tuple(args[k][i] for k in range(len(args)))
            args_i_s, kw_i_s = self.injector.inject(state, args_i, kw_i)
            out = self.model_list[i](*args_i_s, **kw_i_s)
            state = out
            collector.append(out)
            i += 1
        return collector.result()


class ProgressiveWithStateInit(nn.Module, Generic[TLT]):

    def __init__(self,
                 initial: nn.Module,
                 progressive: Progressive[TLT]):
        super().__init__()
        self.initial = initial
        self.progressive = progressive

    def forward(self, *args: List[Tensor], **kw: List[Tensor]) -> TLT:

        kw_i: Dict[str, Tensor] = dict((k, kw[k][0]) for k in kw.keys())
        args_i: Tuple[Tensor, ...] = tuple(args[k][0] for k in range(len(args)))
        state = self.initial(*args_i, **kw_i)

        kw_tail: Dict[str, List[Tensor]] = dict((k, kw[k][1:]) for k in kw.keys())
        args_tail: Tuple[List[Tensor], ...] = tuple(args[k][1:] for k in range(len(args)))

        return self.progressive.forward(state, *args_tail, **kw_tail)


class InputFilter(ABC):
    @abstractmethod
    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        pass


class InputFilterAll(InputFilter):
    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        return args, kw


class InputFilterName(InputFilter):
    def __init__(self, names: Set[str]):
        self.names = names

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_kw = {i: kw[i] for i in self.names}
        return args, filtered_kw


class InputFilterHorisontal(InputFilter):
    def __init__(self, names: Set[str], indices: List[int]):
        self.names = names
        self.indices = indices

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_args = tuple(args[i] for i in self.indices)
        filtered_kw = {i: kw[i] for i in self.names}
        return filtered_args, filtered_kw


class InputFilterVertical(InputFilter):
    def __init__(self, indices: List[int]):
        self.indices = indices

    def filter(self, args: Tuple[List[Tensor], ...], kw: Dict[str, List[Tensor]]):
        filtered_args = tuple([a[j] for j in self.indices] for a in args)
        filtered_kw = {i: [kw[i][j] for j in self.indices] for i in kw.keys()}
        return filtered_args, filtered_kw


class ProgressiveSequential(nn.Module):

    def __init__(self, *modules: Tuple[nn.Module, str, InputFilterHorisontal, InputFilterVertical]):
        super().__init__()
        self.modules = modules

    def forward(self, state: List[Optional[Tensor]], *args: List[Tensor], **kw: List[Tensor]):
        out, slovar = None, {**kw}
        i = 0
        for model, name, horisontal_filter, vertical_filter in self.modules:
            model_args, model_kw = horisontal_filter.filter(args, slovar)
            model_args, model_kw = vertical_filter.filter(model_args, model_kw)
            if state[i] is not None:
                out = model(state[i], *model_args, **model_kw)
            else:
                out = model(*model_args, **model_kw)
            slovar[name] = out
            i += 1
        return out


class ElementwiseModuleList(nn.Module, Generic[TLT]):
    def __init__(self,
                 blocks: List[nn.Module],
                 collector_class: Type[TensorCollector[TLT]] = ListCollector):
        super(ElementwiseModuleList, self).__init__()
        self.model_list = nn.ModuleList(blocks)
        self.collector_class = collector_class

    def forward(self, input: List[Tensor]) -> TLT:
        collector: TensorCollector[TLT] = self.collector_class()
        i = 0
        while i < len(input):
            x = self.model_list[i](input[i])
            collector.append(x)
            i += 1
        return collector.result()


