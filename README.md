# Learning Spatiotemporal Attention for Egocentric Action Recognition

## Overview

This repository contains the PyTorch implementation for the paper "Learning Spatiotemporal Attention for Egocentric Action Recognition", published in the Fifth International Workshop on Egocentric Perception, Interaction and Computing, at ICCV2019.

This code is modifed from the [PyTorch implementation](https://github.com/piergiaj/pytorch-i3d) of the [I3D model](https://arxiv.org/abs/1705.07750), by adding the Spatiotemporal Attention Module (STAM) with gaze supervision for egocentric action recognition, as discribed in the paper.

This code is tested for Pytorch 1.0.0 and python 2.7.14 on Ubuntu 16.04.

## Download

The pre-processed optical flow images, files of gaze locations can be downloaded from the following links.

Google drive: [gaze data](), [flow images]()


Extract the files to the "./dataset/EGTEA_Gaze_plus/" folder.

## Run the evaluation scripts

Example:
```bash
python z_my_test_i3d_flow.py
```
This will finally print results like:
```bash
A
```

## Train the model

Run the bash file [run.sh](run.sh) to train the model:
```bash
./run.sh
```

The .sh file calls the following scipts for training, please choose from 1 of the 4 lines of code.

[z_my_train_i3d_1.py](z_my_train_i3d_1.py) contains the code to fine-tune the I3D for egocentric action recognition, based on the model pre-trained on imagenet and kinetics (see the [original implementation](https://github.com/piergiaj/pytorch-i3d) for details). This code can produce the "RGB-o" and "flow-o" models as discribed in the paper.

[z_my_train_i3d_1_attention_gaze.py](z_my_train_i3d_1_attention_gaze.py) contains the code to train the "RGB-a" and "flow-a" models, which contrain the STAM and uses gaze information as supervision.



