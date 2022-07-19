# Box-Grained Reranking Matching for Multi-Camera Multi-Target Tracking
This project hosts the code of team28 for NVIDIA AI City Challenge 2022 Track 1, City-Scale Multi-Camera Vehicle Tracking. More details can be found in our paper: **Box-Grained Reranking Matching for Multi-Camera Multi-Target Tracking**

Our system obtains the IDF1 at **0.8486** in the public leaderboard, which win the **first place** in the Track1 test set of NVIDIA AI City Challenge 2022.

## Installation
This project is implemented with Python3.6+, [Paddle](https://github.com/PaddlePaddle/Paddle), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

## Data Preparation
### If you want to reproduce our results on the leaderboard of track1. You can download the prepared data below and put them under the corresponding folders.
- Detection
  - Please download the **annotation files, pretrained model and best model** from [this link](https://pan.baidu.com/s/1XQ4iwNHkSaYdcPPEDF1cdw) (The password is ``naif``) 
  - Extract the annotation files under ``DET/swin-transformer-object-detection/data/annotations``
  - Extract the pretrained model under ``DET/swin-transformer-object-detection/pretrain_model``
  - Extract the best model under ``DET/swin-transformer-object-detection/pth``

- REID
  - Please download the **train data**, **model**  and **detection results** from [this link](https://pan.baidu.com/s/1aNB0Q1dhk0hiOV2MRN1Wng) (The password is ``8738``) 
  - Extract the real train data under ``REID/carreid-dynamic-aic22/dataset/train_imgs``
  - Extract the synthetic train data under ``REID/carreid-dynamic-aic22/dataset/syn``
  - Extract the detection results under ``REID/carreid-dynamic-aic22/dataset/crop_test_det_2666_89``
  - Extract the model files under ``REID/carreid-dynamic-aic22/output/``
  
- Tracking
  - Please download the **extracted images of test set by ffmpeg** from [this link](https://pan.baidu.com/s/1H0dfVjj4WjttF0cEbnAg7g ) (The password is ``raw1``) and extract it under ``SCMT/dataset/``.
   ```
  SCMT/dataset/CityFlowV2/AICITY/test/
  ├── c041
    ├── img1
      ├── 0001.jpg
      ├── 0002.jpg
      ...
    ├── seqinfo.ini
  ├── c042
  ├── c043
  ├── c044
  ├── c045
  ├── c046
  ```
  - Please download the **detection results** and **ReID features** from [this link](https://pan.baidu.com/s/1RVY7segBCR3TingcC1UIfA ) (The password is ``56q2``) and extract it under ``SCMT/dataspace/AICITY_test/``.
   ```
  SCMT/dataspace/AICITY_test/
  ├── aic22_1_test_infer_v2_Convnext.pkl
  ├── aic22_1_test_infer_v2_HR48_eps.pkl
  ├── aic22_1_test_infer_v2_R50.pkl
  ├── aic22_1_test_infer_v2_Res2Net200.pkl
  ├── aic22_1_test_infer_v2_ResNext101.pkl
  ├── gen_detection_feat.py
  ```
  
- Matching
  - Please download the **tracking results**, **ReID features** , and **truncation rates** from [this link](https://pan.baidu.com/s/1Qc5rE6OkMaW8vHg-SDdBew  ) (The password is ``0vcw``) and extract it under ``ICA/data/``. All files are listed below.
  ```
  ICA/data/scmt/
  ├── c04*.pkl # ReID features. * indicates 1 to 6
  ├── c04*.txt # Tracking results. * indicates 1 to 6
  ├── c04*_truncation.pkl # truncation rates. * indicates 1 to 6
  ```

  
## Run our system

- Detection
  1. **Generate train/val images:** Run ``python vid2img_recursive_std.py`` in the **TOOLS** folder to generate train/val images for detection.
  2. **Train detection model:** You should first install torch==1.9 in your environment and use our **config** files provided in the **Data Preparation** as the detection configs, then run ``train.sh`` in the **DET/Swin-Transformer-Object-Detection/** folder to train the detection model.
  3. **Test detection model:** Run ``test.sh`` in the **DET/Swin-Transformer-Object-Detection/** folder to get the detection results of test set.

- ReID
  1. **Train ReID model:** Please put all required files in the right place as described inthe **Data Preparation**. Then just run ``train.sh`` in the **REID/carreid-dynamic-aic22/** folder to train the ReID model.
  2. **Test ReID model:** Run ``eval.sh`` in the **REID/carreid-dynamic-aic22/** folder to generate the ReID results for test set. After run ”eval.sh“, two pickle files such as “query_features.pkl” and “gallery_features.pkl” can be obtained. “gallery_features.pkl” contains the real REID features in dictionary form. The key of the dictionary indicates the name of the input image, and the value indicates the corresponding REID feature.

- Tracking

  After you put all provided data in the right place from the **Data Preparation**, you will get the tracking results by runing
```
cd SCMT/dataspace/AICITY_test
python gen_detection_feat.py
cd ../..

python run_aicty.py AICITY test --dir_save $SAVE_DIR
python stat_occlusion_scmt.py $SAVE_DIR
```
&nbsp; &nbsp; &nbsp; &nbsp; The SCMT tracking results will be generated in the specified **SAVE_DIR**. Make sure the **SAVE_DIR** exists.

- Matching
  
  Please follow the **Data Preparation** to put the tracking results and ReID features in the right place and run ``run.sh`` in the **matching** folder to get the final matching result.
