"""
fuse_method:
0 concatenate
1 mean
"""
import os
import pdb
import numpy as np
import pickle
import paddle
import paddle.nn.functional as F

postfix = 'aic22_1_test_infer_v2_'
models = ['HR48_eps', 'ResNext101', 'R50', 'Convnext', 'Res2Net200']
fuse_method = 0

print("input {} models.".format(len(models)))
fuse_fea = {}
for idx in range(len(models)):
    model = models[idx]
    query = pickle.load(open(postfix + model + '.pkl', 'rb'), encoding='latin1')

    for each in query:
        if idx == 0:
            fea = paddle.to_tensor(query[each])
            fuse_fea[each] = F.normalize(fea, axis=0)
        else:
            ori_fea = fuse_fea[each]
            fea = paddle.to_tensor(query[each])
            fea = F.normalize(fea, axis=0)
            if fuse_method == 0:
                fea = paddle.concat(x=[ori_fea, fea], axis=0)
            else:
                fea = ori_fea + fea
            fuse_fea[each] = fea

for each in fuse_fea:
    fea = F.normalize(fuse_fea[each], axis=0)
    fuse_fea[each] = fea.numpy()

with open(postfix + 'fuse.pkl', 'wb') as fid:
    pickle.dump(fuse_fea, fid)
    

