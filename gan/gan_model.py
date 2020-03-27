from itertools import chain
from typing import List, Dict, Callable, Tuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import init

from gan.discriminator import Discriminator
from gan.loss.hinge import HingeLoss
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from models.common import View
from models.lambdaf import LambdaF
from models.munit import MsImageDis, AdaINGen
from models.uptosize import MakeNoise, Uptosize
from munit.networks import StyleEncoder, ContentEncoder, Conv2dBlock
from munit.utils import weights_init
from stylegan2.model import Generator as StyleGenerator2, EqualLinear, ConvLayer
from stylegan2.model import Discriminator as StyleDiscriminator2
from gan.generator import Generator
from gan.loss.gan_loss import GANLoss
from gan.loss_base import Loss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss


def gan_weights_init(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class GANModel:

    def __init__(self, generator: Generator, loss: GANLoss, lr: Tuple[float, float] = (0.0002, 0.0002), do_init_ws=True):
        self.generator = generator
        self.loss = loss
        if do_init_ws:
            self.generator.apply(gan_weights_init)
            self.loss.discriminator.apply(gan_weights_init)
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr[0], lr[1])

    def loss_pair(self, real: List[Tensor], fake: List[Tensor], *noise: Tensor) -> MinMaxLoss:
        return MinMaxLoss(
            self.loss.generator_loss(real, fake),
            self.loss.discriminator_loss_with_penalty(real, fake)
        )

    def generator_loss(self, real: List[Tensor], *noise: Tensor) -> Loss:
        fake = self.generator.forward(*noise)
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return self.loss.generator_loss(real, fake)

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.loss.parameters())

    def forward(self, *noise: Tensor):
        return self.generator.forward(*noise)

    def train(self, real: List[Tensor], *noise: Tensor):
        fake = self.generator.forward(*noise)
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        loss = self.loss_pair(real, fake, *noise)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()


class ConditionalGANModel(GANModel):

    def loss_pair(self, real: List[Tensor], fake: List[Tensor], condition: Tensor, *noise: Tensor) -> MinMaxLoss:
        return MinMaxLoss(
            self.loss.generator_loss(real + [condition], fake + [condition]),
            self.loss.discriminator_loss_with_penalty(real + [condition], fake + [condition])
        )

    def generator_loss(self, real: List[Tensor], condition: Tensor, *noise: Tensor) -> Loss:
        fake = self.generator.forward(condition, *noise)
        condition = condition.detach()
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return self.loss.generator_loss(real + [condition], fake + [condition])

    def forward(self, condition: Tensor, *noise: Tensor):
        return super().forward(*([condition] + [*noise]))

    def train(self, real: List[Tensor], condition: Tensor, *noise: Tensor):
        fake = self.generator.forward(condition, *noise)
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        loss = self.loss_pair(real, fake, condition, *noise)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()


name_to_gan_loss = {
    "hinge": lambda net_d: HingeLoss(net_d),
    "wasserstein": lambda net_d: WassersteinLoss(net_d, penalty_weight=10),
    "vanilla": lambda net_d: DCGANLoss(net_d)
}


class IdentityPreproc(nn.Module):

        def forward(self, style1: Tensor, style2: Tensor):
            noise = None
            return [style1, style2], noise


class StyleGen2Wrapper(Generator):

    def __init__(self,
                 gen: StyleGenerator2,
                 return_latents=False,
                 inject_index=None,
                 truncation=1,
                 truncation_latent=None,
                 input_is_latent=False,
                 randomize_noise=False):
        super().__init__()
        self.gen: StyleGenerator2 = gen
        self.preproc = IdentityPreproc()

        self.return_latents = return_latents
        self.inject_index = inject_index
        self.truncation = truncation
        self.truncation_latent = truncation_latent
        self.input_is_latent = input_is_latent
        self.randomize_noise = randomize_noise

    def forward(self, *input: Tensor):

        styles, noise = self.preproc(*input)
        if not isinstance(styles, list):
            styles = [styles]

        img, latent = self.gen(styles,
                        self.return_latents,
                        self.inject_index,
                        self.truncation,
                        self.truncation_latent,
                        self.input_is_latent,
                        noise,
                        self.randomize_noise)

        if self.return_latents:
            return img, latent
        else:
            return img


class StyleDisc2Wrapper(Discriminator):

    def __init__(self, disc: StyleDiscriminator2):
        super().__init__()
        self.disc = disc
        
    def forward(self, *img: Tensor):
        return self.disc(*img)


def stylegan2(path: str, loss_type: str, lr: float, pretrained_disc=False) -> GANModel:

    state_dict = torch.load(path)

    generator: StyleGenerator2 = StyleGenerator2(256, 512, 8, channel_multiplier=2)
    generator.load_state_dict(state_dict['g'])
    generator = StyleGen2Wrapper(generator).cuda()

    discriminator: StyleDiscriminator2 = StyleDiscriminator2(256, 2)
    if pretrained_disc:
        discriminator.load_state_dict(state_dict['d'])
    discriminator = StyleDisc2Wrapper(discriminator).cuda()

    loss: GANLoss = name_to_gan_loss[loss_type](discriminator)

    return GANModel(generator, loss, lr, do_init_ws=False)


def stylegan2_cond_transfer(path: str, loss_type: str, lr: float, size1: int, size2: int, image_size: int) -> ConditionalGANModel:

    gan_model = stylegan2(path, loss_type, lr, pretrained_disc=False)

    class GanPreproc(nn.Module):
        def __init__(self):
            super().__init__()

            lr_mlp = 0.1

            self.style1 = nn.Sequential(
                # PixelNorm(),
                EqualLinear(size1, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            self.style2 = nn.Sequential(
                # PixelNorm(),
                EqualLinear(size2, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            self.noise = MakeNoise(7, size1)
            # self.noise.apply(gan_weights_init)

        def forward(self, cond: Tensor, z: Tensor):
            noise = self.noise(cond)
            return [self.style1(cond), self.style2(torch.cat([cond, z], dim=1))], noise

    netG = gan_model.generator
    netG.preproc = GanPreproc().cuda()
    netG.inject_index = 2
    netG.input_is_latent = True

    netD = gan_model.loss.discriminator

    netD.disc.convs = nn.Sequential(
        LambdaF(Uptosize(size1, 5, image_size),
                lambda module, img, cond: torch.cat([img, module(cond)], dim=1)),
        ConvLayer(8, 128, 1),
        netD.disc.convs[1:]
    ).cuda()

    loss: GANLoss = name_to_gan_loss[loss_type](netD)

    cond_gan_model = ConditionalGANModel(
        netG,
        loss,
        lr=0.002,
        do_init_ws=False
    )

    params = MinMaxParameters(cond_gan_model.generator.preproc.parameters(),
                              cond_gan_model.loss.discriminator.parameters())

    cond_gan_model.optimizer = MinMaxOptimizer(params, 0.001, 0.001) \
        .add_param_group(
           (cond_gan_model.generator.gen.parameters(), None), (0.0005, None)
        )

    return cond_gan_model


def stylegan2_transfer(path: str, loss_type: str, lr: float, size1: int, size2: int) -> GANModel:

    gan_model = stylegan2(path, loss_type, lr, pretrained_disc=True)

    class GanPreproc(nn.Module):
        def __init__(self):
            super().__init__()

            lr_mlp = 0.05

            self.style1 = nn.Sequential(
                # PixelNorm(),
                EqualLinear(size1, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            self.style2 = nn.Sequential(
                # PixelNorm(),
                EqualLinear(size2, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            self.noise = MakeNoise(7, size1)
            # self.noise.apply(gan_weights_init)

        def forward(self, cond: Tensor, z: Tensor):
            noise = self.noise(cond)
            return [self.style1(cond), self.style2(torch.cat([cond, z], dim=1))], noise

    netG = gan_model.generator
    netG.preproc = GanPreproc().cuda()
    netG.inject_index = 2
    netG.input_is_latent = True

    # netD = gan_model.loss.discriminator

    params = MinMaxParameters(gan_model.generator.preproc.parameters(),
                              gan_model.loss.discriminator.parameters())

    gan_model.optimizer = MinMaxOptimizer(params, 0.001, 0.001) \
        .add_param_group(
           (gan_model.generator.gen.parameters(), None), (0.0005, None)
        )

    return gan_model


class Gen_wrapper(nn.Module):
    def __init__(self, gen, args):
        super().__init__()
        self.increaser = nn.Sequential(
            nn.Linear(140, 64*64),
            View(-1, 8, 8),
            nn.Upsample(scale_factor=2),
            # Conv2dBlock(140, 118, 3, 1, 1, norm='ln', activation=args.activ, pad_type=args.pad_type),
            # nn.Upsample(scale_factor=2),
            # Conv2dBlock(118, 96, 3, 1, 1, norm='ln', activation=args.activ, pad_type=args.pad_type),
            # nn.Upsample(scale_factor=2),
            # Conv2dBlock(512, 256, 3, 1, 1, norm='none', activation=args.activ, pad_type=args.pad_type),
            # nn.Upsample(scale_factor=4),
            Conv2dBlock(64, 128, 3, 1, 1, norm='ln', activation=args.activ, pad_type=args.pad_type),
            nn.Upsample(scale_factor=4),
            Conv2dBlock(128, 64, 3, 1, 1, norm='ln', activation=args.activ, pad_type=args.pad_type),
            # nn.Upsample(scale_factor=4),
            # Conv2dBlock(32, 64, 3, 1, 1, norm='ln', activation=args.activ, pad_type=args.pad_type),
            # nn.Conv2d(16, 64, 3, 1, 1)
        ).cuda()
        self.gen = gen

    def forward(self, cont, style):
        cont = self.increaser(cont)
        return self.gen(cont, style)


def ganmodel_munit(loss_type: str, lr: Tuple[float, float], args) -> GANModel:
    input_dim = args.input_dim
    dim = args.dim
    style_dim = args.style_dim
    n_downsample = args.n_downsample
    n_res = args.n_res
    activ = args.activ
    pad_type = args.pad_type
    mlp_dim = args.mlp_dim
    n_layer = args.n_layer
    norm = args.norm
    num_scales = args.num_scales

    gen = AdaINGen(input_dim, dim, style_dim, n_downsample, n_res, activ, pad_type, mlp_dim).cuda()
    # conv_module = nn.Conv2d(1, 64, 3, 1, 1).cuda()
    # gen = LambdaF(
    #     [conv_module, gen],
    #     lambda cont, style: gen(conv_module(cont), style)
    # )
    gen = Gen_wrapper(gen, args)

    disc = MsImageDis(input_dim, n_layer, dim * 2, norm, activ, num_scales, pad_type).cuda()

    gen.apply(weights_init('kaiming'))
    disc.apply(weights_init('gaussian'))

    loss: GANLoss = name_to_gan_loss[loss_type](disc)
    gan_model = GANModel(
        gen,
        loss,
        lr=lr,
        do_init_ws=False
    )

    return gan_model


def cont_style_munit_enc(args, path: Optional[str] = None):
    enc_style = StyleEncoder(n_downsample=4, input_dim=args.input_dim, dim=args.dim, style_dim=args.style_dim,
                             norm=args.norm, activ=args.activ, pad_type=args.pad_type).cuda()
    enc_content: ContentEncoder = ContentEncoder(args.n_downsample, args.n_res, args.input_dim, args.dim, 'in',
                                                 args.activ,
                                                 args.pad_type).cuda()

    enc_content = nn.Sequential(
        enc_content,
        # nn.Conv2d(64, 1, 3, 1, 1),
        Conv2dBlock(64, 128, 6, 4, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
        Conv2dBlock(128, 64, 4, 2, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
        View(-1),
        nn.Linear(64 * 64, 20),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(20, 140),
        nn.Sigmoid()
        # Conv2dBlock(256, 512, 6, 4, 1, norm='none', activation=args.activ, pad_type=args.pad_type),
        # Conv2dBlock(72, 96, 6, 4, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
        # Conv2dBlock(96, 118, 4, 2, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
        # Conv2dBlock(118, 140, 4, 2, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
    ).cuda()

    enc_style = nn.Sequential(
        enc_style,
        nn.Tanh()
    ).cuda()

    enc_content.apply(weights_init('kaiming'))
    enc_style.apply(weights_init('kaiming'))

    enc = LambdaF([enc_content, enc_style], lambda img: (enc_content.forward(img) * 255/256, enc_style.forward(img)))

    if path:
        print("loading model from " + path)
        enc.load_state_dict(torch.load(path))

    return enc
