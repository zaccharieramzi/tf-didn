# TensorFlow implementation of the Deep iterative down-up CNN

[![Build Status](https://travis-ci.com/zaccharieramzi/tf-didn.svg?branch=master)](https://travis-ci.com/zaccharieramzi/tf-didn)

The Deep iterative down-up CNN (DIDN) is a network introduced by Songhyun Yu et
al. in "Deep Iterative Down-Up CNN for Image Denoising" CVPR 2019.
If you use this network, please cite their work appropriately.

The official implementation is available [here](https://github.com/SonghyunYu/DIDN)
in Pytorch.

The goal of this implementation in TensorFlow is to be easy to read and to adapt:
- all the code is in one file
- defaults are those from the paper
- there is no other imports than from TensorFlow

Some implementation details were taken from the code and not the paper itself:
- no bias is used in the convolutions
- the number of down-up blocks is set to 6
- the activation of the last convolutional layer of the network is linear
