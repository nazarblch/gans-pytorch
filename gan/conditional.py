from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from framework.gan.discriminator import Discriminator
from framework.gan.generator import Generator
from framework.gan.noise.Noise import Noise


class ConditionalGenerator(Generator):

    def __init__(self):
        super(ConditionalGenerator, self).__init__()

    @abstractmethod
    def forward(self, condition: Tensor, *noize: Tensor) -> Tensor: pass


class ConditionalDiscriminator(Discriminator):

    @abstractmethod
    def forward(self, x: Tensor, *condition: Tensor) -> Tensor: pass




