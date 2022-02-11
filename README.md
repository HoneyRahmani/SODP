
## Table of Content
> * [Single Object Detection - Pytorch](#SingleObjectDetection-Pytorche)
>   * [About the Project](#AbouttheProject)
>   * [About Database](#AboutDatabases)
>   * [Built with](#Builtwith)
>   * [Installation](#Installation)
>   * [Examples](#Example)

# Single Object Detection - Fovea Location Detection - Pytorch
## About the Project

In Single Object Detection, the goal is locating only one object in a given image.
This project focuses on developing deep learning model using Pytorch to detect single object. The location of the object will shown by a bounding box.
Bounding box can be depicted with four numbers:

    - [x0, y0, w, h]
    - [x0, y0, x1, y1]
    - [xc, yc, w, h]

x0, y0: The coordinates of the top left of the bounding box

x1, y1: The coordinates of the bottom right of the bounding box

w, h: The width and height of the bounding box

xc, yc: The coordinates of the centroid of the bounding box

In this project, width and hieght is fixed to reduce problem.Thus, task is predicting a bounding box using two number.


![recipe](https://user-images.githubusercontent.com/75105778/153649787-46a34ba4-83b7-4a1f-9e9f-87babf9a3d95.jpg)


## About Database

Find dataset in https://amd.grand-challenge.org 

The database is included two folders, AMD with 89 images and Non-AMD with 311 images, and an Excel file named Fovea_location.xlsx that contains the centroid locations of fovea in images of AMD and None-AMD folders. 

## Built with
* Pytorch
* seaborn
* Model is ResNet…
	![sodModel](https://user-images.githubusercontent.com/75105778/153649873-cc477191-136a-4265-95e5-6524c8a0e5da.jpg)
* smoothed-L1 Loss function.
	For mor informations: https://pytorch.org/docs/stable/nn.html#smoothl1loss
* Intersection over Union (IOU) performance metric.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch
    •	conda install -c anaconda seaborn

## Examples

![2](https://user-images.githubusercontent.com/75105778/153650062-79c6b907-9b35-4660-b168-ab4e2e700447.png)

![3](https://user-images.githubusercontent.com/75105778/153650081-2191b32b-3e98-417e-9b64-fde2c29fbd6b.png)

Location is predicted by Model

![4-deploy](https://user-images.githubusercontent.com/75105778/153650161-2f8b73f0-0069-4149-b526-5c81cf83f455.png)


