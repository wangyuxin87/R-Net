# R-Net

# Description
  This is an implement of "R-Net: A Relationship Network for Efficient and Accurate Scene Text Detection".
  The model is pre-trained on SynthText and achieves 85.6 F-score with 21.1 FPS on ICDAR 2015 Challenge 4.
  
# Requirement

* Python3 
* PyTorch-0.4.1 
* torchvision-0.2.1 
* shapely-1.6.4.post2 
* lanms-1.0.2 
* opencv-python(4.1.0.25)
  
# Installation

## Data

Download data images and annotions from [ICDAR 2015 Challenge 4]:https://rrc.cvc.uab.es/?ch=4&com=downloads. Prepare data as:

~~~
./dataset/train/imgs
./dataset/train/txt
./dataset/test/imgs
./dataset/test/txt
~~~

## Pre-trained Model and Our trained Model.

Download pre-trained [VGG16]:https://drive.google.com/file/d/1HgDuFGd2q77Z6DcUlDEfBZgxeJv4tald/view and [our traind model]:www.baidu.com. Put both models into output/

# Test

CUDA_VISIBLE_DEVICES=0 python test_rnet.py

# Train

CUDA_VISIBLE_DEVICES=0,1 python train_rnet.py

