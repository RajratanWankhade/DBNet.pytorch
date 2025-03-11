# Project Title
A PyTorch re-implementation of Real-time Scene Text Detection with Differentiable Binarization, featuring an updated model trained for 600 epochs and an updated requirements.txt file.

The model was trained on Windows 11 using an NVIDIA RTX 3060 (8GB) GPU, with 300 epochs taking approximately 8 hours to complete.


[Source Link to repo](https://github.com/WenmuZhou/DBNet.pytorch)

## Dataset 

[Click here to access the dataset](https://drive.google.com/drive/folders/1b8s2NZ4MEMj7Zig7kcWHy-yoi1gGeaHz?usp=sharing)

## Changes Made

### `requirement.txt`
- Updated dependencies to the latest versions.
- Added new libraries required for data processing and model training.

## Data Preparation

A new script has been added to the `datasets` directory to prepare data for training. This script handles data loading, preprocessing, and augmentation.

## Run it to executive training :
```sh
python tools/train.py --config_file config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml
```



### Usage

To prepare your data for training, run the following script:

```sh
python datasets/prepare_data.py
```

## Output
![Predicted Image](test/input/img_10.jpg)
![Result Image](test/img_10_result.jpg)

