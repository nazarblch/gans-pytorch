from __future__ import print_function
#%matplotlib inline
import random
import time

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn, Tensor
from torchvision import utils

from gan.dcgan.discriminator import ConvICNN128
from stylegan2.model import Generator as StyleGen
from gan.conjugate_gan_model import ConjugateGANModel
from gan.loss.penalties.conjugate import ConjugateGANLoss, ConjugateGANLoss2
from models.grad import Grad
from models.positive import PosDiscriminator
from stylegan2.train import mixing_noise

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 8
image_size = 256
noise_size = 512

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

dataset = dset.ImageFolder(root="/raid/data/celeba",
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

netG = StyleGen(256, 512, 4, 1).to(device)
netD_1 = PosDiscriminator(256).to(device)
netD_2 = PosDiscriminator(256).to(device)

gan_model = ConjugateGANModel(netG, ConjugateGANLoss2(netD_1, netD_2))

print("Starting Training Loop...")

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)
        noise = mixing_noise(batch_size, 512, 0.9, device)

        loss_d = gan_model.train_disc(imgs, noise)

        loss_g = 0
        if i > 50:
            loss_g = gan_model.train_gen(noise)

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss_d, loss_g))

        if i % 100 == 0:
            # with torch.no_grad():
                fake = gan_model.forward(noise).detach().cpu()
                # print(fake)
                utils.save_image(
                    fake, f'sample_{i}.png', nrow=batch_size//2, normalize=True, range=(-1, 1)
                )
