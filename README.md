# KeypointDETR: an end-to-end 3d keypoint detector

KeypointDETR is accepted to ECCV 2024 as an oral presentation!

## Run

### 1. Extracting Geodesic Distance Maps

[scripts/geodesic_distance.py](https://github.com/bibi547/KeypointDETR/blob/master/scripts/geodesic_distance.py)

Geodesic distance maps are extracted during the data preprocessing phase for generating the ground truth heatmaps. 
Compute the shortest geodesic distance from points to the keypoints and save the results as '.txt' files.

### 2. Config

Modify the config [config/keypoint_saliency.yaml](https://github.com/bibi547/KeypointDETR/blob/master/config/keypoint_saliency.yaml) for your path, filename, and categories.

### 5. Requirement

...

### 6. Train

```
python train.py
```


### 7. Test

```
python test.py
```

## Citation

```
@inproceedings{jin2024keypointdetr,
  title={KeypointDETR: an end-to-end 3d keypoint detector},
  author={Jin, Hairong and Shen, Yuefan and Lou, Jianwen and Zhou, Kun and Zheng, Youyi},
  booktitle={ECCV},
  year={2024}
}
```

## Acknowledgement

[DGCNN](https://github.com/WangYueFt/dgcnn)

[Point Transformer](https://github.com/qq456cvb/Point-Transformers)

[KeypointNet Dataset](https://github.com/qq456cvb/KeypointNet.git)

