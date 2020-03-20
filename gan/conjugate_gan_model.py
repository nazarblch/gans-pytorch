from typing import List, Dict, Callable

import torch
from torch import Tensor, nn
from torch.nn import init
from torch import optim
from gan.discriminator import Discriminator
from gan.gan_model import gan_weights_init
from gan.loss.hinge import HingeLoss
from gan.loss.penalties.conjugate import ConjugateGANLoss
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from stylegan2_pytorch.model import Generator as StyleGenerator2
from stylegan2_pytorch.model import Discriminator as StyleDiscriminator2
from gan.generator import Generator
from gan.loss.gan_loss import GANLoss
from gan.loss_base import Loss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss


class ConjugateGANModel:

    def __init__(self, generator: Generator, loss: ConjugateGANLoss, lr=0.0002, do_init_ws=True):
        self.generator = generator
        self.loss = loss
        if do_init_ws:
            self.generator.apply(gan_weights_init)
            self.loss.Dx.apply(gan_weights_init)
            self.loss.Tyx.apply(gan_weights_init)

        betas = (0.5, 0.999)
        self.g_opt = optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.d_opt = optim.Adam(loss.parameters(), lr=lr, betas=betas)
        self.t_opt = optim.Adam(loss.Tyx.parameters(), lr=lr, betas=betas)

    def generator_loss(self, *noise: Tensor) -> Loss:
        fake = self.generator.forward(*noise)
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return self.loss.generator_loss(fake[0])

    def forward(self, *noise: Tensor):
        fake = self.generator.forward(*noise)
        return fake
        # return self.loss.d_grad(fake)

    def train_disc(self, real: List[Tensor], *noise: Tensor):
        # self.loss.transport_loss(real[0]).maximize_step(self.t_opt)
        fake = self.generator.forward(*noise)
        loss = self.loss.discriminator_loss(fake.detach(), real[0])
        loss.minimize_step(self.d_opt)

        return loss.item()

    def train_gen(self, *noise: Tensor):

        fake = self.generator.forward(*noise)
        loss = self.loss.generator_loss(fake)
        loss.minimize_step(self.g_opt)

        return loss.item()
