# Self-supervised Optical Flow

## Dataset

You can download the dataset for training the FlowNetS network with the script `dataset/download_dataset.sh`. By far it will download the KITTI flow 2015 and KITTI flow 2012 dataset for you. You can also extend this script for other interesting datasets.

## Training

To train on KITTI dataset, one can use `KITTI_occ` or `KITTI_noc` as parameter to `--dataset` in `code/main.py`. One possible full command is:
```
python main.py --dataset KITTI_occ ../dataset/kitti_scene_flow/training/ -b8 -j8 -a flownets
```
