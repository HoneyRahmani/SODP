
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


![recipe](https://user-images.githubusercontent.com/75105778/153648116-60cf980e-8fc8-44da-838c-7a958a5969f2.jpg)


## About Database

Find dataset in https://amd.grand-challenge.org 

Database is included two folder, AMD with 89 image and Non-AMD with 311 image, and a Excel file named Fovea_location.xlsx that contains the centroid locations of fovea in image of AMD and None-AMD folders. 

## Built with
* Pytorch
* seaborn
* Model is ResNet…
	![sodModel](https://user-images.githubusercontent.com/75105778/153648049-92267160-f4d0-46bb-9b67-4b40a695cc89.jpg)
* smoothed-L1 Loss function.
	For mor informations: https://pytorch.org/docs/stable/nn.html#smoothl1loss
* Intersection over Union (IOU) performance metric.

## Installation
    •	conda install pytorch torchvision cudatoolkit=coda version -c pytorch
    •	conda install -c anaconda seaborn

## Example

![2](https://user-images.githubusercontent.com/75105778/153647985-bb74aa61-f4fc-483d-b613-f93a9c3577a7.png)

![3](https://user-images.githubusercontent.com/75105778/153647984-4f13f00f-e40b-4a1a-9be3-94c9fbbf373e.png)


Location is predicted by Model

![4-deploy](https://user-images.githubusercontent.com/75105778/153647555-18339382-58e4-45ba-ae4f-bda5e7423138.png)
