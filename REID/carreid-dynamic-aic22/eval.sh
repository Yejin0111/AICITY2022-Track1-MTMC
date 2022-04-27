#!/usr/bin/env bashcd

# test inference model
export CUDA_VISIBLE_DEVICES=1
python3.7 tools/eval_only_feature.py -c ./ppcls/configs/HR_W48_C_eval.yaml \
                                     -o Global.pretrained_model=output/hr_epoch_50


