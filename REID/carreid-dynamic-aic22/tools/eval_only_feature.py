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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import pdb
import pickle
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppcls.utils import config
from ppcls.engine.engine import Engine

if __name__ == "__main__":
    query_file = './dataset/single.list'
    gallery_file = './dataset/aic22_track1_test_infer_v2.list'
    f1 = open(query_file, 'r')
    f2 = open(gallery_file, 'r')
    query_list = []
    gallery_list = []
    for line in f1.readlines():
        img_name = line.strip()
        query_list.append(img_name)
    for line in f2.readlines():
        img_name = line.strip()
        gallery_list.append(img_name)


    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    engine = Engine(config, mode="eval_feature")
    gallery_feas, query_feas = engine.eval_feature()
    gallery_feas = gallery_feas.numpy()
    query_feas = query_feas.numpy()
    query_dict = {}
    for i in range(len(query_list)):
        img_name = query_list[i]
        query_dict[img_name] = query_feas[i, :]
    with open('query_features.pkl','wb') as fid:
        pickle.dump(query_dict, fid)

    gallery_dict = {}
    for i in range(len(gallery_list)):
        img_name = gallery_list[i]
        gallery_dict[img_name] = gallery_feas[i, :]
    with open('gallery_features.pkl','wb') as fid:
        pickle.dump(gallery_dict, fid)
