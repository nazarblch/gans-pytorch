from itertools import chain
from typing import List, Tuple

import torch

from gan.discriminator import Discriminator
from gan.generator import Generator
from torch import Tensor, nn

from gan.loss.penalties.penalty import default_mix
from gan.loss_base import Loss


class ConjugateGANLoss:

    def __init__(
            self,
            Dx: Discriminator,
            Tyx: Generator,
            pen_weight: float = 10):

        self.Dx = Dx
        self.Tyx = Tyx
        self.pen_weight = pen_weight
        self.mix = default_mix

    def parameters(self):
        return chain(self.Dx.parameters(), self.Tyx.parameters())

    def gradient_point(self, x: List[Tensor], y: List[Tensor]) -> List[Tensor]:
        x0: List[Tensor] = [
            self.mix(xi, yi).detach().requires_grad_(True) for xi, yi in zip(x, y)
        ]
        return x0

    def d_grad(self, x0: Tensor, create_graph=True) -> Tensor:

        if not x0.requires_grad:
            x0.requires_grad_(True)

        dx0: Tensor = self.Dx.forward(x0)

        res = torch.autograd.grad(outputs=dx0,
                                   inputs=x0,
                                   grad_outputs=torch.ones(dx0.shape, device=dx0.device),
                                   create_graph=create_graph,
                                   only_inputs=True)
        return res[0]

    def product(self, x: Tensor, y: Tensor):
        n = x[0].shape[0]
        return (x * y).view(n, -1).sum(1).mean()

    def transport_loss(self, y: Tensor):
        y = y.detach()
        ty: Tensor = self.Tyx(y)

        return Loss(self.product(ty, y) - self.Dx(ty).mean())

    def discriminator_loss(self, x: Tensor, y: Tensor):
        L1 = nn.L1Loss()

        x = x.detach()
        y = y.detach()

        tyx: Tensor = self.Tyx(y).detach()  # detach ?
        loss: Tensor = self.Dx(x).mean() + self.product(tyx, y) - self.Dx(tyx).mean()

        x0 = self.gradient_point([tyx], [x])[0]
        tx0y = self.d_grad(x0)
        x0_pred = self.Tyx(tx0y)

        y0 = self.gradient_point([y], [self.d_grad(x, False)])[0]
        ty0x = self.Tyx(y0)
        y0_pred = self.d_grad(ty0x)

        pen = L1(y0_pred, y0) + L1(x0_pred, x0)

        return Loss(loss + self.pen_weight * pen)

    def generator_loss(self, x: Tensor):

        L1 = nn.L1Loss()
        tx: Tensor = self.d_grad(x, False).detach()

        return Loss(L1(x, tx))



