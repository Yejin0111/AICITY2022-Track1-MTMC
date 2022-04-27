#!/usr/bin/env bash

# for single card train
python3.7 tools/train.py -c ./ppcls/configs/HR_W48_C_tricks.yaml

# for multi-cards train
# python3.7 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ppcls/configs/HR_W48_C_tricks.yaml
