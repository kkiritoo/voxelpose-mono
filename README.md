## Main Results
### Results on MPII val
| Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1|
|---|---|---|---|---|---|---|---|---|---|
| 256x256_pose_resnet_50_d256d256d256 | 96.351 | 95.329 | 88.989 | 83.176 | 88.420 | 83.960 | 79.594 | 88.532 | 33.911 |
| 384x384_pose_resnet_50_d256d256d256 | 96.658 | 95.754 | 89.790 | 84.614 | 88.523 | 84.666 | 79.287 | 89.066 | 38.046 |
| 256x256_pose_resnet_101_d256d256d256 | 96.862 | 95.873 | 89.518 | 84.376 | 88.437 | 84.486 | 80.703 | 89.131 | 34.020 |
| 384x384_pose_resnet_101_d256d256d256 | 96.965 | 95.907 | 90.268 | 85.780 | 89.597 | 85.935 | 82.098 | 90.003 | 38.860 |
| 256x256_pose_resnet_152_d256d256d256 | 97.033 | 95.941 | 90.046 | 84.976 | 89.164 | 85.311 | 81.271 | 89.620 | 35.025 |
| 384x384_pose_resnet_152_d256d256d256 | 96.794 | 95.618 | 90.080 | 86.225 | 89.700 | 86.862 | 82.853 | 90.200 | 39.433 


# VoxelPose

This is the official implementation for:
> [**VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment**](https://arxiv.org/abs/2004.06239),            
> Hanyue Tu, Chunyu Wang, Wenjun Zeng        
> *ECCV 2020 (Oral) ([arXiv 2004.06239](https://arxiv.org/abs/2004.06239))*


<img src="data/panoptic2.gif" width="800"/>


## Installation
1. Clone this repo, and we'll call the directory that you cloned multiview-multiperson-pose as ${POSE_ROOT}.
2. Install dependencies.

## Data preparation

### Shelf/Campus datasets
1. Download the datasets from http://campar.in.tum.de/Chair/MultiHumanPose and extract them under `${POSE_ROOT}/data/Shelf` and `${POSE_ROOT}/data/CampusSeq1`, respectively.

2. We have processed the camera parameters to our formats and you can download them from this repository. They lie in `${POSE_ROOT}/data/Shelf/` and `${POSE_ROOT}/data/CampusSeq1/`,  respectively.

3. Due to the limited and incomplete annotations of the two datasets, we don't train our model using this dataset. Instead, we directly use the 2D pose estimator trained on COCO, and use independent 3D human poses from the Panoptic dataset to train our 3D model. It lies in `${POSE_ROOT}/data/panoptic_training_pose.pkl`. See our paper for more details.

4. For testing, we first estimate 2D poses and generate 2D heatmaps for these two datasets in this repository.  The predicted poses can also download from the repository. They lie in `${POSE_ROOT}/data/Shelf/` and `${POSE_ROOT}/data/CampusSeq1/`,  respectively. You can also use the models trained on COCO dataset (like HigherHRNet) to generate 2D heatmaps directly.

The directory tree should look like this:
```
${POSE_ROOT}
|-- data
    |-- Shelf
    |   |-- Camera0
    |   |-- ...
    |   |-- Camera4
    |   |-- actorsGT.mat
    |   |-- calibration_shelf.json
    |   |-- pred_shelf_maskrcnn_hrnet_coco.pkl
    |-- CampusSeq1
    |   |-- Camera0
    |   |-- Camera1
    |   |-- Camera2
    |   |-- actorsGT.mat
    |   |-- calibration_campus.json
    |   |-- pred_campus_maskrcnn_hrnet_coco.pkl
    |-- panoptic_training_pose.pkl
```


### CMU Panoptic dataset
1. Download the dataset by following the instructions in [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) and extract them under `${POSE_ROOT}/data/panoptic_toolbox/data`.
- You can only download those sequences you need. You can also just download a subset of camera views by specifying the number of views (HD_Video_Number) and changing the camera order in `./scripts/getData.sh`. The sequences and camera views used in our project can be obtained from our paper.
- Note that we only use HD videos,  calibration data, and 3D Body Keypoint in the codes. You can comment out other irrelevant codes such as downloading 3D Face data in `./scripts/getData.sh`.
2. Download the pretrained backbone model from [pretrained backbone](https://1drv.ms/u/s!AjX41AtnTHeTjn3H9PGSLcbSC0bl?e=cw7SQg) and place it here: `${POSE_ROOT}/models/pose_resnet50_panoptic.pth.tar` (ResNet-50 pretrained on COCO dataset and finetuned jointly on Panoptic dataset and MPII).

The directory tree should look like this:
```
${POSE_ROOT}
|-- models
|   |-- pose_resnet50_panoptic.pth.tar
|-- data
    |-- panoptic-toolbox
        |-- data
            |-- 16060224_haggling1
            |   |-- hdImgs
            |   |-- hdvideos
            |   |-- hdPose3d_stage1_coco19
            |   |-- calibration_160224_haggling1.json
            |-- 160226_haggling1  
            |-- ...
```

## Training
### CMU Panoptic dataset

Train and validate on the five selected camera views. You can specify the GPU devices and batch size per GPU  in the config file. We trained our models on two GPUs.
```
python run/train_3d.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml
```
### Shelf/Campus datasets
```
python run/train_3d.py --cfg configs/shelf/prn64_cpn80x80x20.yaml
python run/train_3d.py --cfg configs/campus/prn64_cpn80x80x20.yaml
```

## Evaluation
### CMU Panoptic dataset

Evaluate the models. It will print evaluation results to the screen./
```
python test/evaluate.py --cfg configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml
```
### Shelf/Campus datasets

It will print the PCP results to the screen.
```
python test/evaluate.py --cfg configs/shelf/prn64_cpn80x80x20.yaml
python test/evaluate.py --cfg configs/campus/prn64_cpn80x80x20.yaml
```

## Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{voxelpose,
    author={Tu, Hanyue and Wang, Chunyu and Zeng, Wenjun},
    title={VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020}
}
```
