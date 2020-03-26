"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from typing import List

from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from discriminator import Discriminator
from generator import Generator
from models.common import View
from munit.networks import Conv2dBlock, StyleEncoder, ContentEncoder, Decoder, MLP

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

##################################################################################
# Discriminator
##################################################################################


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
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        next_dim = 0
        for i in range(deep - 2):
            next_dim = min(dim * 2, 256)
            cnn_x += [Conv2dBlock(dim, next_dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = min(dim * 2, 256)
        cnn_x += [View(-1), nn.Linear(next_dim * 4, 1)]
        # cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs: List[torch.Tensor] = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)

        return torch.cat(outputs, dim=0)


##################################################################################
# Generator
##################################################################################

class AdaINGen(Generator):
    # AdaIN auto-encoder architecture
    def __init__(
            self,
            input_dim,
            dim,
            style_dim,
            n_downsample,
            n_res,
            activ,
            pad_type,
            mlp_dim
    ):
        super(AdaINGen, self).__init__()

        # style encoder
        #self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        #self.enc_content: ContentEncoder = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        self.dec: Decoder = Decoder(n_downsample, n_res, dim * 2 ** n_downsample, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)


    def forward(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.reshape(-1)#mean.contiguous().view(-1)
                m.weight = std.reshape(-1)#std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params









