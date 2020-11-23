import random
import time
import os, sys

sys.path.append(os.path.join(sys.path[0], '../..'))
sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '../stylegan2'))

from typing import List
import torch.utils.data
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from dataset.lazy_loader import LazyLoader, Celeba
from gan.loss.stylegan import StyleGANLoss
from gan.models.stylegan import StyleGanModel
from gan.nn.stylegan.discriminator import Discriminator
from gan.nn.stylegan.generator import Generator, FromStyleConditionalGenerator
from gan.noise.stylegan import mixing_noise
from modules.accumulator import Accumulator
from parameters.path import Paths


def send_images_to_tensorboard(writer, data: Tensor, name: str, iter: int, count=8, normalize=True, range=(-1, 1)):
    with torch.no_grad():
        grid = make_grid(
            data[0:count], nrow=count, padding=2, pad_value=0, normalize=normalize, range=range,
            scale_each=False)
        writer.add_image(name, grid, iter)


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 16
image_size = 256
noise_size = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

test_sample_z = torch.randn(8, noise_size, device=device)
Celeba.batch_size = batch_size

generator = Generator(FromStyleConditionalGenerator(image_size, noise_size), n_mlp=8)
generator_ema = Accumulator(Generator(FromStyleConditionalGenerator(image_size, noise_size), n_mlp=8))
generator_ema.storage_model.load_state_dict(generator.state_dict())
generator = generator.cuda()

discriminator = Discriminator(image_size).cuda()

gan_model = StyleGanModel(generator, StyleGANLoss(discriminator), (0.001, 0.0015))

writer = SummaryWriter(f"{Paths.default.board()}/celeba{int(time.time())}")

print("Starting Training Loop...")
starting_model_number = 0

for i in range(100000):

    print(i)

    real_img = next(LazyLoader.celeba().loader).to(device)

    noise: List[Tensor] = mixing_noise(batch_size, noise_size, 0.9, device)
    fake, _ = generator.forward(noise, return_latents=False)

    gan_model.discriminator_train([real_img], [fake.detach()])
    gan_model.generator_loss([real_img], [fake]).minimize_step(gan_model.optimizer.opt_min)

    generator_ema.accumulate(generator, i, 0.98)

    if i % 100 == 0:
        generator_ema.write_to(generator)

    if i % 100 == 0:
        print(i)
        with torch.no_grad():
            fake_test, _ = generator.forward([test_sample_z])
            send_images_to_tensorboard(writer, fake_test, "FAKE", i)


    if i % 10000 == 0 and i > 0:
        torch.save(
            {
                'g': generator.state_dict(),
                'd': discriminator.state_dict(),
                'g_ema': generator_ema.state_dict()
            },
            f'{Paths.default.models()}/celeba_gan_256_{str(i + starting_model_number).zfill(6)}.pt',
        )
