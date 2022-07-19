# coding: utf-8

import os
from subprocess import call
from glob import glob

src_root='data/AIC21_Track3_MTMC_Tracking/train'
des_root='data/AIC21_Track3_MTMC_Tracking_label/train'

#src_root='data/AIC21_Track3_MTMC_Tracking/validation'
#des_root='data/AIC21_Track3_MTMC_Tracking_label/validataion'

if not os.path.isdir(des_root):
    os.makedirs(des_root)

#flist1 = [y for x in os.walk(src_root) for y in glob(os.path.join(x[0], '*.asf'))]
flist2 = [y for x in os.walk(src_root) for y in glob(os.path.join(x[0], '**.avi'))]
vid_list = flist2
print(len(vid_list))

for v in vid_list:
    vp = v
    fbase, fext = os.path.splitext(vp[(len(src_root) + 1):]) 
    fbase="/".join(fbase.split("/")[:-1]) 
    dp = os.path.join(des_root, fbase)
    name="_".join(fbase.split("/")[-2:]) 
    if not os.path.isdir(dp):
        os.makedirs(dp)
    call(['sh', 'vid2img_std.sh', vp, dp, str(1), name])
    
