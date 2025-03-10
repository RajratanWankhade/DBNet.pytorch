# Real-time Scene Text Detection with Differentiable Binarization

### My Setup

- Windows 11 running on Nvidia RTX 3060 GPU 8GB Ram 
- For 300 Epochs it took around 8 Hours 
- For whole 1200 Epochs: Author said takes around a day.  

## Changes Made

### `requirement.txt`
- Updated dependencies to the latest versions.
- Added new libraries required for data processing and model training.

## Data Preparation

A new script has been added to the `datasets` directory to prepare data for training. 




### Usage

To prepare your data for training, run the following script:

```sh
python datasets/prepare_data.py  