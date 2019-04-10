# Simple Baselines for Human Pose Estimation and Tracking

## News
- Our new work [High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514) is available at [HRNet](https://github.com/HRNet). Our HRNet has been applied to a wide range of vision tasks, such as [image classification](https://github.com/HRNet/HRNet-Image-Classification), [objection detection](https://github.com/HRNet/HRNet-Object-Detection), [semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) and [facial landmark](https://github.com/HRNet/HRNet-Facial-Landmark-Detection).
- Our new work [Deep High-Resolution Representation Learning for Human Pose Estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) has already been released at <https://github.com/leoxiaobin/deep-high-resolution-net.pytorch>. The best single HRNet can obtain an **AP of 77.0** on COCO test-dev2017 dataset and **92.3% of PCKh@0.5** on MPII test set. The new repositoty also support the SimpleBaseline method, and you are welcomed to try it.<br>
- Our entry using this repo has won the winner of [PoseTrack2018 Multi-person Pose Tracking Challenge](https://posetrack.net/workshops/eccv2018/posetrack_eccv_2018_results.html)!<br>
- Our entry using this repo ranked 2nd place in the [keypoint detection task of COCO 2018](http://cocodataset.org/#keypoints-leaderboard)!

## Introduction
This is an official pytorch implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). This work provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field. State-of-the-art results are achieved on challenging benchmarks. On COCO keypoints valid dataset, our best **single model** achieves **74.3 of mAP**. You can reproduce our results using this repo. All models are provided for research purpose.    </br>

## Main Results
### Results on MPII val
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1|
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 96.351 | 95.329 | 88.989 | 83.176 | 88.420 | 83.960 | 79.594 | 88.532 | 33.911 |
| 384x384_pose_resnet_50_d256d256d256 | 96.658 | 95.754 | 89.790 | 84.614 | 88.523 | 84.666 | 79.287 | 89.066 | 38.046 |
| 256x256_pose_resnet_101_d256d256d256 | 96.862 | 95.873 | 89.518 | 84.376 | 88.437 | 84.486 | 80.703 | 89.131 | 34.020 |
| 384x384_pose_resnet_101_d256d256d256 | 96.965 | 95.907 | 90.268 | 85.780 | 89.597 | 85.935 | 82.098 | 90.003 | 38.860 |
| 256x256_pose_resnet_152_d256d256d256 | 97.033 | 95.941 | 90.046 | 84.976 | 89.164 | 85.311 | 81.271 | 89.620 | 35.025 |
| 384x384_pose_resnet_152_d256d256d256 | 96.794 | 95.618 | 90.080 | 86.225 | 89.700 | 86.862 | 82.853 | 90.200 | 39.433 |

### Note:
- Flip test is used.

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x192_pose_resnet_50_d256d256d256 | 0.704 | 0.886 | 0.783 | 0.671 | 0.772 | 0.763 | 0.929 | 0.834 | 0.721 | 0.824 |
| 384x288_pose_resnet_50_d256d256d256 | 0.722 | 0.893 | 0.789 | 0.681 | 0.797 | 0.776 | 0.932 | 0.838 | 0.728 | 0.846 |
| 256x192_pose_resnet_101_d256d256d256 | 0.714 | 0.893 | 0.793 | 0.681 | 0.781 | 0.771 | 0.934 | 0.840 | 0.730 | 0.832 |
| 384x288_pose_resnet_101_d256d256d256 | 0.736 | 0.896 | 0.803 | 0.699 | 0.811 | 0.791 | 0.936 | 0.851 | 0.745 | 0.858 |
| 256x192_pose_resnet_152_d256d256d256 | 0.720 | 0.893 | 0.798 | 0.687 | 0.789 | 0.778 | 0.934 | 0.846 | 0.736 | 0.839 |
| 384x288_pose_resnet_152_d256d256d256 | 0.743 | 0.896 | 0.811 | 0.705 | 0.816 | 0.797 | 0.937 | 0.858 | 0.751 | 0.863 |


### Results on *Caffe-style* ResNet
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x192_pose_resnet_50_*caffe*_d256d256d256 | 0.704 | 0.914 | 0.782 | 0.677 | 0.744 | 0.735 | 0.921 | 0.805 | 0.704 | 0.783 |
| 256x192_pose_resnet_101_*caffe*_d256d256d256 | 0.720 | 0.915 | 0.803 | 0.693 | 0.764 | 0.753 | 0.928 | 0.821 | 0.720 | 0.802 |
| 256x192_pose_resnet_152_*caffe*_d256d256d256 | 0.728 | 0.925 | 0.804 | 0.702 | 0.766 | 0.760 | 0.931 | 0.828 | 0.729 | 0.806 |


### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- Difference between *PyTorch-style* and *Caffe-style* ResNet is the position of stride=2 convolution

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Disable cudnn for batch_norm:
   ```
   # PYTORCH=/path/to/pytorch
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed  and then set an environment variable (PYTORCH in this case) accordingly.
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
3. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo) and caffe-style pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1yJMSFOnmzwhA4YYQS71Uy7X1Kl_xq9fN?usp=sharing). 
4. Download mpii and coco pretrained models from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW0D5ZE4ArK9wk_fvw) or [GoogleDrive](https://drive.google.com/drive/folders/13_wJ6nC7my1KKouMkQMqyr9r1ZnLnukP?usp=sharing). Please download them under ${POSE_ROOT}/models/pytorch, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet50-caffe.pth.tar
            |   |-- resnet101-5d3b4d8f.pth
            |   |-- resnet101-caffe.pth.tar
            |   |-- resnet152-b121ed2d.pth
            |   `-- resnet152-caffe.pth.tar
            |-- pose_coco
            |   |-- pose_resnet_101_256x192.pth.tar
            |   |-- pose_resnet_101_384x288.pth.tar
            |   |-- pose_resnet_152_256x192.pth.tar
            |   |-- pose_resnet_152_384x288.pth.tar
            |   |-- pose_resnet_50_256x192.pth.tar
            |   `-- pose_resnet_50_384x288.pth.tar
            `-- pose_mpii
                |-- pose_resnet_101_256x256.pth.tar
                |-- pose_resnet_101_384x384.pth.tar
                |-- pose_resnet_152_256x256.pth.tar
                |-- pose_resnet_152_384x384.pth.tar
                |-- pose_resnet_50_256x256.pth.tar
                `-- pose_resnet_50_384x384.pth.tar

   ```

4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Valid on MPII using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### Training on MPII

```
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```

### Valid on COCO val2017 using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar
```

### Training on COCO train2017

```
python pose_estimation/train.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```

### Other Implementations
- TensorFlow [[Version1](https://github.com/mks0601/TF-SimpleHumanPose)]
- PaddlePaddle [[Version1](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/human_pose_estimation)]
- Gluon [[Version1](https://gluon-cv.mxnet.io/model_zoo/pose.html)]

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
