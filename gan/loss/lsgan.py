from typing import List, Callable, Iterator
import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from gan.discriminator import Discriminator
from gan.loss.base import GANLoss
from gan.loss.loss_base import Loss

from gan.loss.penalties.style_gan_penalty import StyleDiscriminatorPenalty, PenaltyWithCounter


class LsGANLoss(GANLoss):

    __criterion = nn.MSELoss()

    def __init__(self, discriminator: Discriminator):
        super().__init__(discriminator=discriminator)

    def _generator_loss(self, dgz: Tensor, real: List[Tensor], fake: List[Tensor]) -> Loss:
        batch_size = dgz.size(0)
        nc = dgz.size(1)

        real_labels = torch.full((batch_size, nc,), 1, device=dgz.device)
        errG = self.__criterion(dgz.view(batch_size, nc).sigmoid(), real_labels)
        return Loss(errG)

    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:
        batch_size = dx.size(0)
        nc = dx.size(1)

        real_labels = torch.full((batch_size, nc,), 1, device=dx.device)
        err_real = self.__criterion(dx.view(batch_size, nc).sigmoid(), real_labels)

        fake_labels = torch.full((batch_size, nc,), 0, device=dx.device)
        err_fake = self.__criterion(dy.view(batch_size, nc).sigmoid(), fake_labels)

        return Loss(-(err_fake + err_real))
