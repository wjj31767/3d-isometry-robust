# PointCloud Saliency Maps (pytorch)

This is pytorch implementation for paper "PointCloud Saliency Maps" http://arxiv.org/abs/1812.01687 (To appear in ICCV 2019 oral presentation, 187 out of 4303)

this is the link of tensorflow version
[tianzheng4](https://github.com/tianzheng4)/**[PointCloud-Saliency-Maps](https://github.com/tianzheng4/PointCloud-Saliency-Maps)**

large copy from [3d-isometry-robust]/**(https://github.com/skywalker6174/3d-isometry-robust)**

# Pytorch Implementation of PointNet and PointNet++ 


## Classification
### Data Preparation
[ModelNet40] automatically downloaded

[ShapeNet] /fxia22/pointnet.pytorch (follow the guidence for downloading)
The default path of data is '/data'.

## train

```
python train.py
```

need train first

## Adversarial attack

```
python random_drop.py
python saliency.py
python critical_drop.py
```

## RESULTS

Random drop 100 points, the result didn't change

Saliency drop 100 points, original point cloud number is 1024. (if point cloud number is 2048, still get the same result)

| Before drop         | After drop         |
| ------------------- | ------------------ |
| 84.279% (2080/2468) | 68.355%(1687/2468) |

Critical drop 100 points, original point cloud number is 1024.

| Before drop         | After drop         |
| ------------------- | ------------------ |
| 84.279% (2080/2468) | 68.436%(1689/2468) |

## Reference By

[3d-isometry-robust]/**(https://github.com/skywalker6174/3d-isometry-robust)**

[tianzheng4](https://github.com/tianzheng4)/**[PointCloud-Saliency-Maps](https://github.com/tianzheng4/PointCloud-Saliency-Maps)**

## Environments
### Ubuntu 

fine with 8GB graphic card

### Windows 10

fine with 8GB graphic card
