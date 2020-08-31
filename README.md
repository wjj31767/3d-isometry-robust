# PointCloud Saliency Maps (pytorch)

This is pytorch implementation for paper "PointCloud Saliency Maps" (To appear in ICCV 2019 oral presentation, 187 out of 4303)

this is the link of tensorflow version
[tianzheng4](https://github.com/tianzheng4)/**[PointCloud-Saliency-Maps](https://github.com/tianzheng4/PointCloud-Saliency-Maps)**

large copy from [3d-isometry-robust]/**(https://github.com/skywalker6174/3d-isometry-robust)**

# Pytorch Implementation of PointNet and PointNet++ 


## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

```
# batch_size=192 if use 12GB GPU
```

### Run
```
## Check model in ./models 
## E.g. pointnet2_msg
python3 train_cls.py --model pointnet_cls --log_dir pointnet_cls
python3 test_cls.py --log_dir pointnet_cls
```

## Adversarial attack

```
python saliency.py
python critical_drop.py
```




## Reference By
[3d-isometry-robust]/**(https://github.com/skywalker6174/3d-isometry-robust)**

[tianzheng4](https://github.com/tianzheng4)/**[PointCloud-Saliency-Maps](https://github.com/tianzheng4/PointCloud-Saliency-Maps)**

## Environments
### Ubuntu
### Windows 10
