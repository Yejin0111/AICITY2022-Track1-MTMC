## *** Requirements ***
paddlepaddle>=2.0
python>=3.7

## *** Train ***
Please download the cropped images (real data, i.e., train and validation data from Track1 of AICity Challenge 2022, and synthetic data).

sh train.sh

Note: Configuration files for 5 models are as follows:
HRNet: ./ppcls/configs/HR_W48_C_tricks.yaml
ConvNext: ./ppcls/configs/ConvNext_tricks.yaml
Res2Net200: ./ppcls/configs/Res2Net200_vd_tricks.yaml
ResNet50: ./ppcls/configs/ResNet50_vd_tricks.yaml
ResNeXt101: ./ppcls/configs/ResNeXt101_32x8d_wsl_tricks.yaml

## *** Extract REID features ***
sh eval.sh

After run ”eval.sh“, two pickle files such as “query_features.pkl” and “gallery_features.pkl” can be obtained. “gallery_features.pkl” contains the real REID features in dictionary form. The key of the dictionary indicates the name of the input image, and the value indicates the corresponding REID feature.

Note: Configuration files for 5 models are as follows:
HRNet: ./ppcls/configs/HR_W48_C_eval.yaml
ConvNext: ./ppcls/configs/ConvNext_eval.yaml
Res2Net200: ./ppcls/configs/Res2Net200_vd_eval.yaml
ResNet50: ./ppcls/configs/ResNet50_vd_eval.yaml
ResNeXt101: ./ppcls/configs/ResNeXt101_32x8d_wsl_eval.yaml

## *** Extracted REID features for SCMT and ICA ***
features/