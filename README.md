
## Table of Content
# Single Object Detection - Pytorch
## About the Project

In Single Object Detection,the goal is locating only one object in given image.
This project focus on developing deep learning model to using Pytorch to detect single object.The location of object will show by a bounding box.
Bounding box can be depected with four numbers:

    - [x0, y0, w, h]
    - [x0, y0, x1, y1]
    - [xc, yc, w, h]

x0, y0: The coordinates of the top left of the bounding box

x1, y1: The coordinates of the bottom right of the bounding box

w, h: The width and height of the bounding box

xc, yc: The coordinates of the centroid of the bounding box

In this project, width and hieght is fixed to reduce problem.Thus, task is predicting a bounding box using two number.

![Steps](https://pasteboard.co/IGe7cI2utgOH.jpg)

## About Database

Find dataset in https://amd.grand-challenge.org 

Database is included two folder, AMD with 89 image and Non-AMD with 311 image, and a Excel file named Fovea_location.xlsx that contains the centroid locations of fovea in image of AMD and None-AMD folders. 

## Built with
* Pytorch
* seaborn
* Model is ResNet…
	![Model](https://pasteboard.co/f9l2o2b1078N.jpg)
* smoothed-L1 Loss function.
	For mor informations: https://pytorch.org/docs/stable/nn.html#smoothl1loss
* Intersection over Union (IOU) performance metric.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch
    •	conda install -c anaconda seaborn

## Example

![Example 1](https://pasteboard.co/icoFzEmBhZgG.png)
![Example 2](https://pasteboard.co/B12kNjEQDpXk.png)

![Location is predicted by Model](https://pasteboard.co/6PBDADJmPECp.png)
