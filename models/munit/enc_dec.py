from typing import List, Tuple
from torch import nn, Tensor
import torch
from gan.discriminator import Discriminator
from gan.generator import Generator
from stylegan2.model import EqualLinear, ResBlock, EqualConv2d, StyledConv, ToRGB, PixelNorm
from models.common import View
from models.stylegan import ModulatedResBlocks, ModulatedResBlock
from models.munit.base import Conv2dBlock, ResBlocksMunit, MLP
from models.uptosize import Uptosize
from stylegan2.model import EqualConv2d

#
# class ContentTensor:
#     def __init__(self, tensor: Tensor):
#         self.tensor = tensor
#
#
# class StyleTensor:
#     def __init__(self, tensor: Tensor):
#         self.tensor = tensor
#
#
# class ImageTensor:
#     def __init__(self, tensor: Tensor):
#         self.tensor = tensor
#
#
# class NoiseTensor:
#     def __init__(self, tensor: Tensor):
#         self.tensor = tensor


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocksMunit(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class StyleEncoder(nn.Module):
    def __init__(self, style_dim):
        super(StyleEncoder, self).__init__()
        self.model = [
            EqualConv2d(3, 16, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            EqualConv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1),
            EqualLinear(256 * 4 * 4, style_dim * 2, activation="fused_lrelu"),
            EqualLinear(style_dim * 2, style_dim * 2),
            View(2, style_dim)
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


##################################################################################
# Typing
##################################################################################


##################################################################################
# Discriminator
##################################################################################

class CondMsImageDis(Discriminator):
    # Multi-scale discriminator architecture
    def __init__(
            self,
            input_dim,
            n_layer,
            dim,
            norm,
            activ,
            num_scales,
            pad_type):
        super(CondMsImageDis, self).__init__()
        self.n_layer = n_layer
        self.dim = dim
        self.norm = norm
        self.activ = activ
        self.num_scales = num_scales
        self.pad_type = pad_type
        self.input_dim = input_dim + 1
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns: nn.ModuleList = nn.ModuleList()
        self.uptosize = Uptosize(140, 1, 256)

        self.channel_size = {
            'res_block_1': (3, 16), # eps x 64
            'res_block_2': (16, 32), # 0.25 x 64
            'res_block_3': (32, 64), # 1 x 64
            'res_block_4': (64, 128), # 4 x 64
            'res_block_5': (128, 256), # 16 x 64
            'res_block_6': (256, 512), # 64 x 64
            # sum 64 + 16 + 4 + 1 + 0.25 + eps
        }


        for i in range(self.num_scales):
            self.cnns.append(self._make_net(n_layer - i))


    def _make_net(self, deep):
        dim = self.dim
        cnn_x = []
        # cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [ResBlock(self.input_dim, dim)]
        next_dim = 0
        for i in range(deep - 3):
            next_dim = min(dim * 2, 512)
            cnn_x += [ResBlock(dim, next_dim)]
            # cnn_x += [Conv2dBlock(dim, next_dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = min(dim * 2, 512)
        cnn_x += [View(-1),
                  EqualLinear(next_dim * 16, next_dim, activation='fused_lrelu'),
                  EqualLinear(next_dim, 1)
                  ]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x: Tensor, condition: Tensor):
        outputs: List[torch.Tensor] = []
        cond = self.uptosize(condition)
        x = torch.cat((x, cond), dim=1)
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return torch.cat(outputs, dim=0)


class MsImageDis(Discriminator):
    # Multi-scale discriminator architecture
    def __init__(
            self,
            input_dim,
            n_layer,
            dim,
            norm,
            activ,
            num_scales,
            pad_type):
        super(MsImageDis, self).__init__()
        self.n_layer = n_layer
        self.dim = dim
        self.norm = norm
        self.activ = activ
        self.num_scales = num_scales
        self.pad_type = pad_type
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_scales):
            self.cnns.append(self._make_net(n_layer - i))

    def _make_net(self, deep):
        dim = self.dim
        cnn_x = []
        # cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [ResBlock(self.input_dim, dim)]
        next_dim = 0
        for i in range(deep - 3):
            next_dim = min(dim * 2, 512)
            cnn_x += [ResBlock(dim, next_dim)]
            # cnn_x += [Conv2dBlock(dim, next_dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = min(dim * 2, 512)
        cnn_x += [View(-1),
                  EqualLinear(next_dim * 16, next_dim, activation='fused_lrelu'),
                  EqualLinear(next_dim, 1)
                  ]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x: Tensor):
        outputs: List[torch.Tensor] = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)

        return torch.cat(outputs, dim=0)


##################################################################################
# Generator
##################################################################################

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim):
        super(Decoder, self).__init__()

        style_dim = 256
        self.n_upsample = n_upsample
        self.channel_size = {
            'increser_conv': (256, 256), # 16 x 64
            'styleconv_1': (256, 128), # 8 x 64
            'styleconv_2': (128, 64), # 2 x 64
            'styleconv_3': (64, 64), # 1 x 64
            'res_block': 64, # 8 x 64
            'styleconv_4': (64, 32), # 0.5 x 64
            'styleconv_5': (32, 16) # 0.25 x 64
            # sum 35.75
        }

        self.increaser = nn.Sequential(
            EqualLinear(140, 256 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, 4, 4),
            nn.Upsample(scale_factor=2),
            EqualConv2d(*self.channel_size['increser_conv'], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.noise1 = nn.Parameter(torch.randn(1, 1, 16, 16), requires_grad=False)
        self.noise2 = nn.Parameter(torch.randn(1, 1, 32, 32), requires_grad=False)
        self.noise3 = nn.Parameter(torch.randn(1, 1, 64, 64), requires_grad=False)
        self.StyleConv1 = StyledConv(*self.channel_size['styleconv_1'], 3, style_dim, upsample=True)
        self.StyleConv2 = StyledConv(*self.channel_size['styleconv_2'], 3, style_dim, upsample=True)
        self.StyleConv3 = StyledConv(*self.channel_size['styleconv_3'], 3, style_dim, upsample=True)

        self.adain_model = ModulatedResBlocks(n_res, self.channel_size['res_block'], style_dim)
        self.to_rgb_1 = ToRGB(dim, style_dim, upsample=False)
        self.to_rgbs = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        self.noise4 = nn.Parameter(torch.randn(1, 1, 128, 128), requires_grad=False)
        self.noise5 = nn.Parameter(torch.randn(1, 1, 256, 256), requires_grad=False)

        self.upsample_noises = [self.noise4, self.noise5]

        # upsampling blocks
        for i in range(n_upsample):
            self.to_rgbs.append(ToRGB(dim//2, style_dim))
            self.upsamples.append(StyledConv(dim, dim // 2, 3, style_dim, upsample=True))
            dim //= 2

    def forward(self, cont: Tensor, style: Tensor)-> Tensor:

        res = self.increaser(cont)

        res = self.StyleConv1(res, style, self.noise1)
        res = self.StyleConv2(res, style, self.noise2)
        res = self.StyleConv3(res, style, self.noise3)

        res = self.adain_model(res, style)
        img = self.to_rgb_1(res, style)

        for i in range(self.n_upsample):
            res = self.upsamples[i](res, style, self.upsample_noises[i])
            # res = self.upsamples[i](res, style)
            img = self.to_rgbs[i](res, style, img)

        return img


class DecoderRGBStart(nn.Module):
    def __init__(self, n_upsample, n_res, dim):
        super(DecoderRGBStart, self).__init__()

        style_dim = 256
        self.n_upsample = n_upsample
        self.channel_size = {
            'increser_conv': (256, 256), # 16 x 64
            'styleconv_1': (256, 128), # 8 x 64
            'styleconv_1_1': (128, 128), # 8 x 64
            'styleconv_2': (128, 64), # 2 x 64
            'styleconv_3': (64, 64), # 1 x 64
            'res_block': 64, # 8 x 64
            'styleconv_4': (64, 32), # 0.5 x 64
            'styleconv_5': (32, 16) # 0.25 x 64
            # sum 35.75
        }
        self.toRGB_8 = ToRGB(256, style_dim, upsample=False)
        self.toRGB_16 = ToRGB(128, style_dim)
        self.toRGB_32 = ToRGB(64, style_dim)
        self.toRGB_64 = ToRGB(64, style_dim)


        self.increaser = nn.Sequential(
            EqualLinear(140, 256 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, 4, 4),
            nn.Upsample(scale_factor=2),
            EqualConv2d(*self.channel_size['increser_conv'], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.noise1 = nn.Parameter(torch.randn(1, 1, 16, 16), requires_grad=False)
        self.noise2 = nn.Parameter(torch.randn(1, 1, 32, 32), requires_grad=False)
        self.noise3 = nn.Parameter(torch.randn(1, 1, 64, 64), requires_grad=False)
        self.StyleConv1 = StyledConv(*self.channel_size['styleconv_1'], 3, style_dim, upsample=True)
        self.StyleConv1_1 = StyledConv(*self.channel_size['styleconv_1_1'], 3, style_dim)
        self.StyleConv2 = StyledConv(*self.channel_size['styleconv_2'], 3, style_dim, upsample=True)
        self.StyleConv2_1 = StyledConv(*self.channel_size['styleconv_3'], 3, style_dim)
        self.StyleConv3 = StyledConv(*self.channel_size['styleconv_3'], 3, style_dim, upsample=True)
        self.StyleConv3_1 = StyledConv(*self.channel_size['styleconv_3'], 3, style_dim)

        self.to_rgbs = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # self.noise4 = nn.Parameter(torch.randn(1, 1, 128, 128), requires_grad=False)
        # self.noise5 = nn.Parameter(torch.randn(1, 1, 256, 256), requires_grad=False)

        # self.upsample_noises = [self.noise4, self.noise5]

        # upsampling blocks
        for i in range(n_upsample):
            self.to_rgbs.append(ToRGB(dim//2, style_dim))
            self.upsamples.append(StyledConv(dim, dim // 2, 3, style_dim, upsample=True))
            dim //= 2

    def forward(self, cont: Tensor, style: Tensor)-> Tensor:

        res = self.increaser(cont)
        image = self.toRGB_8(res, style)
        res = self.StyleConv1(res, style, self.noise1)
        res = self.StyleConv1_1(res, style, self.noise1)
        image = self.toRGB_16(res, style, image)
        res = self.StyleConv2(res, style, self.noise2)
        res = self.StyleConv2_1(res, style, self.noise2)
        image = self.toRGB_32(res, style, image)
        res = self.StyleConv3(res, style)
        res = self.StyleConv3_1(res, style)
        image = self.toRGB_64(res, style, image)

        # res = self.adain_model(res, style)
        # img = self.toRGB_64(res, style, image)

        for i in range(self.n_upsample):
            # res = self.upsamples[i](res, style, self.upsample_noises[i])
            res = self.upsamples[i](res, style)
            image = self.to_rgbs[i](res, style, image)

        return image

class AdaINGen(Generator):
    # AdaIN auto-encoder architecture
    def __init__(
            self,
            input_dim,
            dim,
            style_dim,
            n_downsample,
            n_res,
            mlp_dim
    ):
        super(AdaINGen, self).__init__()

        self.dec: DecoderRGBStart = DecoderRGBStart(n_downsample, n_res, dim * 2 ** n_downsample)
        self.style = MLP(style_dim, style_dim, mlp_dim, 3)

    def forward(self, content: Tensor, style: Tensor)-> Tensor:
        images = self.dec(content, self.style(style))
        return images


class MunitEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc_style = StyleEncoder(style_dim=args.style_dim)
        self.enc_content: ContentEncoder = ContentEncoder(args.n_downsample, args.n_res, args.input_dim, args.dim, 'in',
                                                     args.activ,
                                                     args.pad_type)

        self.style_dim = args.style_dim

        self.enc_content = nn.Sequential(
                self.enc_content,
                Conv2dBlock(64, 128, 6, 4, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
                Conv2dBlock(128, 64, 4, 2, 1, norm='in', activation=args.activ, pad_type=args.pad_type),
                View(-1),
                nn.Linear(64 * 64, 140),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(140, 140),
                nn.Sigmoid()
            )

    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        content = self.enc_content(img) * 255 / 256
        style = self.enc_style(img).view(img.shape[0], self.style_dim)
        return content, style
