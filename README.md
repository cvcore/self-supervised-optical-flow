# Self-supervised Learning of Optical Flow

Collecting groundtruth data for optical flow is hard.

In this research project, we compare and analyze a set of self-supervised losses to train an optical flow network without the groundtruth labels. This approach enables a network to learn optical flow from only pairs of consecutive images or from videos.

The network is able to achieve a validation endpoint error (EPE) of 5.5 on the FlyingChairs dataset, trained with only photometric and smoothness loss. Pretrained weights can be downloaded for evaluation.

![input_sample](code/images/input_2.gif)

![flow_sample](code/images/GT_2.png)

## Setup

To setup, simply clone this repository locally and make sure you have the following packages installed:

- [Anaconda](https://www.anaconda.com)
- [PyTorch](https://pytorch.org), together with torchvision
- [WandB](https://www.wandb.com)
- [Numpy](https://numpy.org)

## Dataset

You can download the dataset for training the FlowNetS network with the script `dataset/download_dataset.sh`.

## Training

To train the model, run

    python code/main.py PATH_DATASET --dataset flying_chairs --arch flownets --device cuda:0

Then, the training log can be seen in `tensorboard` by running:

    tensorboard --logdir flying_chairs/ --host 0.0.0.0

In addition, this script supports training FlownetS and PWCNet with a combination of the following losses:

- Photometric loss
- Smoothness loss
- Forward & backward loss
- Tenary loss
- SSIM loss

Change `get_default_config()` function in `code/main.py` to set weights for each loss.

## Evaluation

For evaluation you can download our pretrained model and run

    python code/run_inference.py PATH_DATASET PATH_MODEL_PTH --output PATH_OUTPUT

Then, the model prediction together with the groundtruth label will be saved in `PATH_OUTPUT` folder.
