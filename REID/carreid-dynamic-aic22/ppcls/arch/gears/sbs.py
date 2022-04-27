"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Uniform
import math

class GemPooling(nn.Layer):
    """
    GemPooling
    """
    def __init__(self, norm=3.0, ks=7):
        super(GemPooling, self).__init__()
        self.norm = norm
        self.ks = ks
        self.avg_pool = nn.AvgPool2D(kernel_size=ks, stride=ks, padding=0, ceil_mode=True)
    
    def forward(self, input):
        """
        GemPooling_forward
        """
        input = paddle.clip(input, min=1e-6, max=1e8)
        input = paddle.pow(input, self.norm)
        input = self.avg_pool(input)
        out = paddle.pow(input, 1.0 / self.norm)
        return out


class SBS(nn.Layer):
    """
    SBS
    """
    def __init__(self, embedding_size, num_features, class_num, kernel_size=7, use_gp=True, lr_mult=1.0):
        super(SBS, self).__init__()
        self.embedding_size = embedding_size
        self.num_features = num_features
        self.class_num = class_num
        self.kernel_size = kernel_size
        self.use_gp = use_gp

        self.gp = GemPooling(ks=self.kernel_size)
        self.bn1 = nn.BatchNorm(
            self.embedding_size,
            param_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
            data_layout="NCHW")
        self.flatten = nn.Flatten()
        stdv1 = 1.0 / math.sqrt(self.embedding_size * 1.0)
        self.fc1 = paddle.nn.Linear(
            self.embedding_size, 
            self.num_features, 
            weight_attr=ParamAttr(initializer=Uniform(-stdv1, stdv1)))
        self.bn2 = nn.BatchNorm(
            self.num_features,
            param_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
            data_layout="NCHW")
        stdv2 = 1.0 / math.sqrt(self.num_features * 1.0)
        self.fc2 = paddle.nn.Linear(
            self.num_features, 
            self.class_num, 
            weight_attr=ParamAttr(initializer=Uniform(-stdv2, stdv2)))
    
    def forward(self, input, label=None):
        """
        SBS_forward
        """
        if self.use_gp:
            input = self.gp(input)
            input = self.bn1(input)
            input = self.flatten(input)
        input = self.fc1(input)      
        reid_fea = self.bn2(input)
        cls_out = self.fc2(reid_fea)
        return cls_out, reid_fea
