"""
# Copyright 2021 Baidu
# Written by jiangminyue on January 12, 2022
# Code was based on https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pdb
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math
from .vision_transformer import trunc_normal_, zeros_, ones_

def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = (x / keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        forward
        """
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Layer):
    """
    Identity
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        forward
        """
        return x

class LayerNorm(nn.Layer):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=ones_
        )
        self.add_parameter("weight", self.weight)
        self.bias = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=zeros_
        )
        self.add_parameter("bias", self.bias)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )


    def forward(self, x):
        """
        forward
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:   
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            #s = x.var(axis=1, unbiased=False, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.eps)
            x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
                + self.bias.unsqueeze(-1).unsqueeze(-1)
            return x

class Block(nn.Layer):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_prob=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = Conv2D(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = Linear(4 * dim, dim)
        self.gamma = self.create_parameter(shape=[dim], default_initializer=Constant(layer_scale_init_value)) 
        self.add_parameter("gamma", self.gamma)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else Identity()

    def forward(self, x):
        """
        forward
        """
        input = x
        x = self.dwconv(x)
        x = x.transpose(perm=[0, 2, 3, 1]) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(perm=[0, 3, 1, 2]) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Layer):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        class_num (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, class_num=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.class_num = class_num

        self.downsample_layers = nn.LayerList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            Conv2D(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    Conv2D(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.LayerList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_prob=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = Linear(dims[-1], class_num)

        self.apply(self._init_weights)

        self.fc = Linear(
            dims[-1], self.class_num)

    def _init_weights(self, m):
        if isinstance(m, (Conv2D, Linear)):
            trunc_normal_(m.weight)
            zeros_(m.bias)

    def forward_features(self, x):
        """
        forward
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(paddle.mean(x, axis=[-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(self, x):
        """
        forward
        """
        x = self.forward_features(x)
        # x = self.fc(x)
        return x



def convnext_tiny(pretrained=False, **kwargs):
    """
    convnext
    """
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext_small(pretrained=False, **kwargs):
    """
    convnext
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnext_base(pretrained=False, **kwargs):
    """
    convnext
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnext_large(pretrained=False, **kwargs):
    """
    convnext
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnext_xlarge(pretrained=False, **kwargs):
    """
    convnext
    """
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model
