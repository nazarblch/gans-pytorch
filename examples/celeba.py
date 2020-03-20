from __future__ import print_function
#%matplotlib inline
import random
import time

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn, Tensor
from torchvision import utils

from gan.conjugate_gan_model import ConjugateGANModel
from gan.dcgan.discriminator import DCDiscriminator, PosDCDiscriminator, ConvICNN128
from gan.dcgan.generator import DCGenerator
from gan.gan_model import stylegan2
from gan.image2image.residual_generator import ResidualGenerator
from gan.loss.penalties.conjugate import ConjugateGANLoss
from gan.noise.normal import NormalNoise
from models.grad import Grad

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 4
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

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=12)

stylegan = stylegan2(
    "/home/ibespalov/stylegan2/stylegan2-pytorch/checkpoint/790000.pt",
    "hinge",
    0.002
)

class GanPreproc(nn.Module):

        def forward(self, z: Tensor):
            return [z, z], None

device = torch.device("cuda")
noise = NormalNoise(noise_size, device)
netG = stylegan.generator
netG.preproc = GanPreproc()
netD = PosDCDiscriminator(image_size).to(device)
netT = Grad(PosDCDiscriminator(image_size)).to(device)


gan_model = ConjugateGANModel(netG, ConjugateGANLoss(netD, netT), lr=0.0012, do_init_ws=False)

# netD.convexify()
# netT.net.convexify()


print("Starting Training Loop...")

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)
        z = noise.sample(batch_size)

        loss_d = gan_model.train_disc([imgs], z)
        # netD.convexify()
        # netT.net.convexify()

        loss_g = 0
        if i % 5 == 0 and i > 0:
            loss_g = gan_model.train_gen(z)

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss_d, loss_g))

        if i % 100 == 0:
            # with torch.no_grad():
                fake = gan_model.forward(z).detach().cpu()
                utils.save_image(
                    fake, f'sample_{i}.png', nrow=batch_size//2, normalize=True, range=(-1, 1)
                )
