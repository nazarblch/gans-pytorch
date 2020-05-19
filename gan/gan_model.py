from itertools import chain
from typing import List, Dict, Callable, Tuple, Optional, TypeVar, Generic, Any

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init

from gan.discriminator import Discriminator
from gan.loss.hinge import HingeLoss
from gan.loss.penalties.style_gan_penalty import StyleGeneratorPenalty, PenaltyWithCounter
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from models.common import View
from models.lambdamodule import LambdaModule
from models.munit.enc_dec import MsImageDis, AdaINGen, MunitEncoder, CondMsImageDis
from models.stylegan import ScaledConvTranspose2d
from models.uptosize import MakeNoise, Uptosize
from models.munit.enc_dec import StyleEncoder, ContentEncoder, Conv2dBlock #LayerNorm
from munit.utils import weights_init
from stylegan2.model import Generator as StyleGenerator2, EqualLinear, ConvLayer, EqualConv2d, StyledConv, ResBlock
from stylegan2.model import Discriminator as StyleDiscriminator2
from gan.generator import Generator
from gan.loss.gan_loss import GANLoss, StyleGANLoss
from gan.loss_base import Loss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def g_path_regularize(fake_img, latents, cond, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=[latents, cond], create_graph=True, only_inputs=True
    )
    path_lengths = torch.sqrt(grad[0].pow(2).sum(2).mean(1) + grad[1].pow(2).sum(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths



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

    def model_save(self, path):
        torch.save({'G': self.generator.state_dict(), 'D': self.loss.discriminator.state_dict()}, path)

    def model_load(self, path):
        weights = torch.load(path)
        self.generator.load_state_dict(weights['G'])
        self.loss.discriminator.load_state_dict(weights['D'])
        return self


class ConditionalGANModel(GANModel):

    def loss_pair(self, real: List[Tensor], fake: List[Tensor], condition: Tensor) -> MinMaxLoss:
        condition = condition.detach()
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
        loss = self.loss_pair(real, fake, condition)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()


name_to_gan_loss = {
    "hinge": lambda net_d: HingeLoss(net_d),
    "wasserstein": lambda net_d: WassersteinLoss(net_d, penalty_weight=10),
    "vanilla": lambda net_d: DCGANLoss(net_d)
}


GeneratorClass = TypeVar('GeneratorClass', bound=Generator)
class CondStyleGanModel(ConditionalGANModel, Generic[GeneratorClass]):

    def __init__(self, generator: GeneratorClass, loss: StyleGANLoss, lr: Tuple[float, float] = (0.0015, 0.002)):
        self.generator = generator
        self.loss = loss
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr[0], lr[1], min_betas=(0, 0.792), max_betas=(0, 0.932))

        self.g_reg_every = 5
        self.path_regularize = 1

        self.gen_penalty = PenaltyWithCounter(
            StyleGeneratorPenalty(self.path_regularize * self.g_reg_every),
            lambda i: i % self.g_reg_every == 0
        )

    def loss_pair(self, real: List[Tensor], fake: List[Tensor], latent: List[Tensor], condition: Tensor) -> MinMaxLoss:
        #condition = condition.detach().requires_grad_(True)
        condition_detach = condition.detach()

        return MinMaxLoss(
            self.loss.generator_loss(real + [condition_detach], fake + [condition_detach]) + self.gen_penalty(fake[0], [latent[0], condition]),
            self.loss.discriminator_loss_with_penalty(real + [condition_detach], fake + [condition_detach])
        )

    def disc_train(self, real: List[Tensor], fake: List[Tensor], condition: Tensor):
        requires_grad(self.loss.discriminator, True)
        # condition_detach = condition.detach()
        # fake = [f.detach() for f in fake]
        self.loss.discriminator_loss_with_penalty(real + [condition], fake + [condition]).maximize_step(
            self.optimizer.opt_max
        )

    def generator_loss(self, real: List[Tensor], fake: List[Tensor], latent: List[Tensor], condition: Tensor) -> Loss:
        requires_grad(self.loss.discriminator, False)
        gen_loss = self.loss.generator_loss(real + [condition], fake + [condition])
        # pen = self.gen_penalty(fake[0], [latent[0], condition])
        return gen_loss


    def train(self, real: List[Tensor], condition: Tensor, noise: List[Tensor]):

        condition = condition.detach().requires_grad_(True)
        condition_detach = condition.detach()

        # requires_grad(self.generator, False)
        requires_grad(self.loss.discriminator, True)

        fake_img, latent = self.generator(condition, noise, return_latents=True)

        self.loss.discriminator_loss_with_penalty(
            real + [condition_detach],
            [fake_img.detach(), condition_detach]).maximize_step(self.optimizer.opt_max)

        requires_grad(self.generator, True)
        requires_grad(self.loss.discriminator, False)

        (
            self.loss.generator_loss(real + [condition_detach], [fake_img, condition_detach]) +\
            self.gen_penalty(fake_img, [latent, condition])
        ).minimize_step(self.optimizer.opt_min)


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
                 input_is_latent=True,
                 randomize_noise=True):
        super().__init__()
        self.gen: StyleGenerator2 = gen
        self.preproc = IdentityPreproc()

        self.return_latents = return_latents
        self.inject_index = inject_index
        self.truncation = truncation
        self.truncation_latent = truncation_latent
        self.input_is_latent = input_is_latent
        self.randomize_noise = randomize_noise

    def decoder(self, cond, latent1):
        noise = self.preproc.noise(cond)
        img, _ = self.gen([latent1, latent1],
                               self.return_latents,
                               self.inject_index,
                               self.truncation,
                               self.truncation_latent,
                               self.input_is_latent,
                               noise,
                               self.randomize_noise)
        return img

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


class CondStyleDisc2Wrapper(Discriminator):

    def __init__(self, disc: StyleDiscriminator2):
        super().__init__()
        self.disc = disc

        self.cond_uptosize = Uptosize(140, 10, 256)

        self.final_linear = nn.Sequential(
            EqualLinear(self.disc.channels[4] * 4 * 4 + 140, self.disc.channels[4], activation='fused_lrelu'),
            EqualLinear(self.disc.channels[4], self.disc.channels[4], activation='fused_lrelu'),
            EqualLinear(self.disc.channels[4], 1),
        )

        self.disc.convs = nn.Sequential(
            ConvLayer(13, 64, 1),
            self.disc.convs[1:]
        )

    def forward_disc(self, input, cond):
        out = self.disc.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.disc.stddev_group)
        stddev = out.view(
            group, -1, self.disc.stddev_feat, channel // self.disc.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.disc.final_conv(out)

        out = out.view(batch, -1)
        out_norm = torch.sqrt(torch.sum(out ** 2)).detach()
        cond_norm = torch.sqrt(torch.sum(cond ** 2)).detach()
        out = torch.cat([out, cond * out_norm / cond_norm], dim = 1)
        out = self.final_linear(out)

        return out

    def forward(self, img: Tensor, cond: Tensor):
        cond_true_size = self.cond_uptosize(cond)
        x = torch.cat([img, cond_true_size], dim=1)

        return self.forward_disc(x, cond) #self.disc(x)


def stylegan2(path: str, loss_type: str, lr: Tuple[float, float], pretrained_disc=False) -> GANModel:

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


def stylegan2_cond_transfer(loss_type: str, lr: Tuple[float,float], image_size: int) -> ConditionalGANModel:

    generator: StyleGenerator2 = StyleGenerator2(256, 512, 8, channel_multiplier=1)
    generator = StyleGen2Wrapper(generator).cuda()

    discriminator: StyleDiscriminator2 = StyleDiscriminator2(256, 1)
    discriminator = CondStyleDisc2Wrapper(discriminator).cuda()

    class GanPreproc(nn.Module):
        def __init__(self):
            super().__init__()

            lr_mlp = 1

            self.style1 = nn.Sequential(
                # PixelNorm(),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            # self.style2 = nn.Sequential(
            #     # PixelNorm(),
            #     EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
            #     EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
            #     EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            self.noise = MakeNoise(7, 140)
            # self.noise.apply(gan_weights_init)

        def forward(self, cond: Tensor, z: Tensor):
            noise = self.noise(cond)
            for i in range(len(noise)):
                if i > 1 and i % 2 == 0:
                    noise[i] = None
            return [self.style1(z), self.style1(z)], noise

    generator.preproc = GanPreproc().cuda()
    generator.inject_index = 2
    generator.input_is_latent = True

    loss: GANLoss = name_to_gan_loss[loss_type](discriminator)

    cond_gan_model = ConditionalGANModel(
        generator,
        loss,
        lr=lr,
        do_init_ws=False
    )

    # params = MinMaxParameters(cond_gan_model.generator.preproc.parameters(),
    #                           cond_gan_model.loss.discriminator.parameters())
    #
    # cond_gan_model.optimizer = MinMaxOptimizer(params, 0.001, 0.001) \
    #     .add_param_group(
    #        (cond_gan_model.generator.gen.parameters(), None), (0.0005, None)
    #     )

    return cond_gan_model


def stylegan2_transfer(loss_type: str, lr: Tuple[float, float]) -> GANModel:

    generator: StyleGenerator2 = StyleGenerator2(256, 512, 8, channel_multiplier=1)
    generator = StyleGen2Wrapper(generator).cuda()

    discriminator: StyleDiscriminator2 = StyleDiscriminator2(256, 1)
    discriminator = StyleDisc2Wrapper(discriminator).cuda()

    class GanPreproc(nn.Module):
        def __init__(self):
            super().__init__()

            lr_mlp = 1

            self.style1 = nn.Sequential(
                # PixelNorm(),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
                EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            # self.style2 = nn.Sequential(
            #     # PixelNorm(),
            #     EqualLinear(size2, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
            #     EqualLinear(256, 256, lr_mul=lr_mlp, activation='fused_lrelu'),
            #     EqualLinear(256, 512, lr_mul=lr_mlp, activation='fused_lrelu'))

            # self.noise = MakeNoise(7, 140)

        def forward(self, z: Tensor):
            return [self.style1(z)], None

    generator.preproc = GanPreproc().cuda()
    # generator.inject_index = 2
    generator.input_is_latent = True

    loss: GANLoss = name_to_gan_loss[loss_type](discriminator)

    gan_model = GANModel(
        generator,
        loss,
        lr=lr,
        do_init_ws=False
    )

    return gan_model



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

    gen = AdaINGen(input_dim, dim, style_dim, n_downsample, n_res, mlp_dim).cuda()
    disc = MsImageDis(input_dim, n_layer, dim, norm, activ, num_scales, pad_type).cuda()
    # disc: StyleDiscriminator2 = StyleDiscriminator2(256, 2).cuda()

    loss: GANLoss = name_to_gan_loss[loss_type](disc)
    gan_model = GANModel(
        gen,
        loss,
        lr=lr,
        do_init_ws=False
    )

    return gan_model


def cont_style_munit_enc(args, cont_path: Optional[str] = None, style_path: Optional[str] = None):
    enc = MunitEncoder(args)
    if cont_path:
        print("loading model from " + cont_path)
        enc.enc_content.load_state_dict(torch.load(cont_path, map_location='cpu'))
        # enc.enc_content.load_state_dict(torch.load(cont_path))

    if style_path:
        print("loading model from " + style_path)
        enc.enc_style.load_state_dict(torch.load(style_path))

    return enc.cuda()


def cond_ganmodel_munit(args, munit_args, path, starting_model_number: int):
    cont_style_encoder: MunitEncoder = cont_style_munit_enc(
        munit_args,
        None,  # "/home/ibespalov/pomoika/munit_content_encoder15.pt",
        None  # "/home/ibespalov/pomoika/munit_style_encoder_1.pt"
    )  # .to(device)

    args.latent = 512
    args.n_mlp = 5

    args.start_iter = 0

    generator = CondGen2(StyleGenerator2(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ))  # .to(device)

    discriminator = CondStyleDisc2Wrapper(StyleDiscriminator2(
        args.size, channel_multiplier=args.channel_multiplier
    ))  # .to(device)
    weights = torch.load(f"{path}stylegan2_invertable_{str(starting_model_number).zfill(6)}.pt",
                         map_location="cpu")
    generator.load_state_dict(weights['g'])
    generator = generator.cuda()
    discriminator.load_state_dict(weights['d'])
    discriminator = discriminator.cuda()
    cont_style_encoder.load_state_dict(weights['enc'])
    cont_style_encoder = cont_style_encoder.cuda()

    return generator, discriminator, cont_style_encoder




class CondGen2(nn.Module):

    def __init__(self, gen: Generator):
        super().__init__()

        self.gen: Generator = gen

        self.noise = MakeNoise(7, 140, [512, 512, 512, 512, 256, 128, 64])

        self.condition_preproc = nn.Sequential(
            EqualLinear(140, 256 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, 4, 4),
            EqualConv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.inject_index = 2

    def decode(self, cond: Tensor, latent: Tensor):
        noise = self.noise(cond)
        input = self.condition_preproc(cond)
        latent = [latent[:, 0], latent[:, 1]]
        return self.gen(latent, condition=input, noise=noise, input_is_latent=True, inject_index=self.inject_index)[0]

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):

        noise = self.noise(cond)
        input = self.condition_preproc(cond)

        # for i in range(len(noise)):
        #     if i > 1 and i % 2 == 0:
        #         noise[i] = None

        return self.gen(z, condition=input, noise=noise, return_latents=return_latents, inject_index=self.inject_index)


class CondGen3(nn.Module):

    def __init__(self, gen: StyleGenerator2):
        super().__init__()

        self.gen: Generator = gen
        self.noise_up = nn.ModuleList([
            ScaledConvTranspose2d(68, self.gen.channels[128], 3),
            ScaledConvTranspose2d(self.gen.channels[128], self.gen.channels[256], 3),
        ])
        self.noise_down = nn.ModuleList([
            ConvLayer(68, self.gen.channels[64], 3, downsample=False),
            ConvLayer(self.gen.channels[64], self.gen.channels[32], 3, downsample=True),
            ConvLayer(self.gen.channels[32], self.gen.channels[16], 3, downsample=True),
            ConvLayer(self.gen.channels[16], self.gen.channels[8], 3, downsample=True),
            ConvLayer(self.gen.channels[8], self.gen.channels[4], 3, downsample=True)
        ])
        self.inject_index = 2

    def make_noise(self, heatmap: Tensor):
        x = heatmap * 10
        noise_up_list = []
        for i in self.noise_up:
            x = i.forward(x)
            noise_up_list.append(x)
            noise_up_list.append(x)

        y = heatmap * 10
        noise_down_list = []
        for i in self.noise_down:
            y = i.forward(y)
            noise_down_list.append(y)
            noise_down_list.append(y)

        return noise_down_list[-2::-1] + noise_up_list

    def decode(self, cond: Tensor, latent: Tensor):
        noise = self.make_noise(cond)
        latent = [latent[:, 0], latent[:, 1]]
        return self.gen(latent, condition=noise[0], noise=noise, input_is_latent=True, inject_index=self.inject_index)[0]

    def forward(self, cond: Tensor, z: List[Tensor], return_latents=False):
        noise = self.make_noise(cond)
        return self.gen(z, condition=noise[0], noise=noise, return_latents=return_latents, inject_index=self.inject_index)


class CondDisc3(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 256,
            32: 256,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.channels = channels
        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 6, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        in_channel = in_channel + 68

        convs_with_hm = []
        for i in range(6, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs_with_hm.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs_with_hm = nn.Sequential(*convs_with_hm)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input, cond):
        out = self.convs(input)
        out = torch.cat((out, cond * 10), dim=1)
        out = self.convs_with_hm(out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out