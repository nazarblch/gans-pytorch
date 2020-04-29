from typing import List, Dict, Callable

import torch
from torch import Tensor, nn
from torch.nn import init
from torch import optim
from gan.discriminator import Discriminator
from gan.gan_model import gan_weights_init
from gan.loss.hinge import HingeLoss
from gan.loss.penalties.conjugate import ConjugateGANLoss, ConjugateGANLoss2
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from gan.generator import Generator
from gan.loss.gan_loss import GANLoss
from gan.loss_base import Loss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss


class StyleConjugateGANModel:

    def __init__(self, generator: Generator, loss: ConjugateGANLoss2):
        self.generator = generator
        self.loss = loss

        self.g_opt = optim.Adam(generator.parameters(), lr=0.0015, betas=(0, 0.792))
        self.d_opt = optim.Adam(loss.parameters(), lr=0.005, betas=(0, 0.932))

    def generator_loss(self, noise: List[Tensor]) -> Loss:
        fake, _ = self.generator.forward(noise)
        return self.loss.generator_loss(fake)

    def forward(self, noise: List[Tensor]):
        fake, _ = self.generator.forward(noise)
        return fake
        # return self.loss.d_grad(fake)

    def train_disc(self, real: Tensor, noise: List[Tensor]):
        fake, _ = self.generator.forward(noise)
        loss = self.loss.discriminator_loss(fake.detach(), real)
        loss.minimize_step(self.d_opt)

        return loss.item()

    def train_gen(self, noise: List[Tensor]):
        fake, _ = self.generator.forward(noise)
        loss = self.loss.generator_loss(fake)
        loss.minimize_step(self.g_opt)

        return loss.item()

class ConjugateGANModel:

    def __init__(self, generator: Generator, loss: ConjugateGANLoss2):
        self.generator = generator
        self.loss = loss

        self.g_opt = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.792))
        self.d_opt = optim.Adam(loss.parameters(), lr=0.001, betas=(0.5, 0.932))

    def generator_loss(self, noise: Tensor) -> Loss:
        fake = self.generator.forward(noise)
        return self.loss.generator_loss(fake)

    def forward(self, noise: Tensor):
        fake = self.generator.forward(noise)
        return fake
        # return self.loss.d_grad(fake)

    def train_disc(self, real: Tensor, noise: Tensor):
        fake = self.generator.forward(noise)
        loss = self.loss.discriminator_loss(fake.detach(), real)
        loss.minimize_step(self.d_opt)

        return loss.item()

    def train_gen(self, noise: Tensor):
        fake = self.generator.forward(noise)
        loss = self.loss.generator_loss(fake)
        loss.minimize_step(self.g_opt)

        return loss.item()
