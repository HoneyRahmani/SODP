# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:57:50 2020

@author: asus
"""

import numpy as np
path2data = ".\\data\\"
# =======================================
# Functions
# Load images and labels
def load_img_label(labels_df, id_):
    
    imgName = labels_df["imgName"]
    if imgName [id_][0] == "A":
        perfix = "AMD"
    else :
        perfix = "Non_AMD"
    fullPath2img = os.path.join(path2data,"Training400",
                                perfix,imgName[id_])
    img = Image.open(fullPath2img)
    x = labels_df["Fovea_X"][id_]
    y = labels_df["Fovea_Y"][id_]
    
    label = (x,y)
    return img, label

# Show images and labels
def show_img_label(img, label, w_h=(50,50),thickness=2):
    
    w,h = w_h
    cx,cy = label
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([(cx-w/2, cy-h/2), (cx+w/2,cy+h/2)],outline="green",width=thickness)
    plt.imshow(np.asarray(img))
# Preprocess Function
def resize_img_label(img, label = (0.0,0.0), target_size = (256,256)):
    
    w_orig, h_orig = img.size
    w_target, h_target = target_size
    
    cx,cy = label
    image_new = TF.resize(img,target_size)
    label_new = cx/w_orig*w_target, cy/h_orig*h_target
    
    return image_new,label_new

def random_hflip(image,label):
    w,h=image.size
    x,y=label        

    image = TF.hflip(image)
    label = w-x, y
    return image,label

def random_vflip(image,label):
    w,h=image.size
    x,y=label

    image = TF.vflip(image)
    label = x, h-y
    return image, label

def random_shift(image,label,max_translate=(0.2,0.2)):
    w,h=image.size
    max_t_w, max_t_h=max_translate
    cx, cy=label

    # translate coeficinet, random [-1,1]
    trans_coef=np.random.rand()*2-1
    w_t = int(trans_coef*max_t_w*w)
    h_t = int(trans_coef*max_t_h*h)

    image=TF.affine(image,translate=(w_t, h_t),shear=0,angle=0,scale=1)
    label = cx+w_t, cy+h_t
        
    return image,label

def transformer(image,label, params):

    image, label = resize_img_label(image,label,params["target_size"])
    
    if random.random() < params["p_hflip"]:
        image, label = random_hflip(image, label)
    
    if random.random() < params["p_vflip"]:
        image, label = random_vflip(image, label)
    
    if random.random() < params["p_shift"]:
        image, label = random_shift(image, label, params["max_translate"])
    
    image = TF.to_tensor(image) 
    return image, label

def rescale_label(a,b):
    div = [ai*bi for ai,bi in zip(a,b)]
    return div

def show(img,label=None):
    npimg = img.numpy().transpose((1,2,0))
    #plt.axis([0,30000,0,30000])
    if label is not None:
        label=rescale_label(label,img.shape[1:])        
        x,y=label
    plt.imshow(npimg)
    plt.plot(x,y,'b+',markersize=20)
    plt.show()
    
def scale_label(a,b):
    div = [ai/bi for ai,bi in zip(a,b)]
    return div
   
# =======================================
# Exploring data
import os
import pandas as pd


path2labels = os.path.join(path2data, "Training400","Fovea_location.xlsx" )

labels_df = pd.read_excel(path2labels, index_col = "ID")
# In order to show head function output in "spyder" IDE should be use "print"
# In order to show tail function output in "notebook" IDE can be use without "print"
print(labels_df.head())
print(labels_df.tail())

#Select a random set of image ids
nrows, ncolmn = 2,3
imgName = labels_df["imgName"]
ids = labels_df.index
rndIds = np.random.choice(ids, nrows*ncolmn)
print(rndIds)
#### The scatter plot of the Fovea_X and Fovea_Y
import seaborn as sns
AorN = [imn[0] for imn in labels_df.imgName]
sns.scatterplot(x=labels_df['Fovea_X'], y=labels_df.Fovea_Y, hue=AorN)
#### Show a few sample images
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

np.random.seed(2019)

plt.rcParams['figure.figsize'] = (15,9)
plt.subplots_adjust(wspace=0, hspace=0.3)
nrows,ncols = 2,3

imgName = labels_df["imgName"]
# genrate random
#ids = np.random.randint(0,len(imgName), nrows*ncols)
ids = labels_df.index
rndIds = np.random.choice(ids, nrows*ncols)
print(ids)
  
for i, id_ in enumerate(rndIds):
    
    img,label = load_img_label(labels_df,id_)
    print(img.size,label)
    plt.subplot(nrows,ncols,i+1)
    show_img_label(img, label, w_h=(150,150),thickness=20)
    plt.title(imgName[id_])    
# =========================================
# Plot the distributions of heights and widths
import torchvision.transforms.functional as F

h_list, w_list = [],[]

for id_ in ids:
    
    if imgName[id_][0] == "A":
        prefix ="AMD"
    else:
        prefix ="Non_AMD"
    fullPath2img = os.path.join(path2data,"Training400",prefix,imgName[id_])
    img = Image.open(fullPath2img)
    h,w = img.size
    h_list.append(h)
    w_list.append(w)
sns.displot(a=h_list, kde=False)
sns.distplot(a=h_list, kde=False)
sns.distplot(a=w_list, kde=False)

# ============================================
# Preprocessing
import random
import torchvision.transforms.functional as TF

np.random.seed(0)
random.seed(0)

image, label = load_img_label(labels_df,1)
params = {
        "target_size": (256,256),
        "p_hflip": 1.0,
        "p_vflip": 1.0,
        "p_shift": 1.0,
        "max_translate": (0.2,0.2),
        }

img_t, label_t = transformer(image, label,params)

plt.subplot(1,2,1)
show_img_label(image, label, w_h=(150,150),thickness=20)
plt.subplot(1,2,2)
show_img_label(TF.to_pil_image(img_t), label_t)
#######################
# load image and label
img, label=load_img_label(labels_df,1)   

# resize image and label
img_r,label_r=resize_img_label(img,label)

# adjust brightness
img_t=TF.adjust_brightness(img_r,brightness_factor=0.5)
label_t=label_r

plt.subplot(1,2,1)
show_img_label(img_r,label_r)
plt.subplot(1,2,2)
show_img_label(img_t,label_t)

# brightness
img_t=TF.adjust_contrast(img_r,contrast_factor=0.4)

# gamma correction
img_t=TF.adjust_gamma(img_r,gamma=1.4)

# ===============================================
# Custom dataset

from torch.utils.data import Dataset
from PIL import Image

class AMD_dataset(Dataset):
    def __init__(self, path2data, transform, trans_params):      
        pass    
      
    def __len__(self):
        # return size of dataset
        return len(self.labels)
      
    def __getitem__(self, idx):
        pass
   
    
def __init__(self, path2data, transform, trans_param):
        
        path2labels=os.path.join(path2data,"Training400","Fovea_location.xlsx")
        
        labels_df = pd.read_excel(path2labels, index_col="ID")
        self.labels = labels_df[["Fovea_X","Fovea_Y"]].values
        
        self.imgName = labels_df["imgName"]
        self.ids = labels_df.index
        
        self.fullPath2image = [0]*len(self.ids)
        
        for id_ in self.ids:
            
            if self.imgName[id_][0]=="A":
                prefix ="AMD"
            else:
                prefix ="Non_AMD"
            
            self.fullPath2image[id_-1]=os.path.join(path2data,"Training400",prefix,self.imgName[id_])
        self.transform = transform
        self.trans_params = trans_param
            
    
def __getitem__(self, idx):
        
       image = Image.open(self.fullPath2image[idx])
       label = self.labels[idx]
       image, label = self.transform(image, label, self.trans_params)
       
       return image, label
   
AMD_dataset.__init__=__init__
AMD_dataset.__getitem__=__getitem__
###################
np.random.seed(1)
def transformer(image, label, params):
    image,label=resize_img_label(image,label,params["target_size"])

    if random.random() < params["p_hflip"]:
        image,label=random_hflip(image,label)
        
    if random.random() < params["p_vflip"]:            
        image,label=random_vflip(image,label)
        
    if random.random() < params["p_shift"]:                            
        image,label=random_shift(image,label, params["max_translate"])

    if random.random() < params["p_brightness"]:
        brightness_factor=1+(np.random.rand()*2-1)*params["brightness_factor"]
        image=TF.adjust_brightness(image,brightness_factor)

    if random.random() < params["p_contrast"]:
        contrast_factor=1+(np.random.rand()*2-1)*params["contrast_factor"]
        image=TF.adjust_contrast(image,contrast_factor)

    if random.random() < params["p_gamma"]:
        gamma=1+(np.random.rand()*2-1)*params["gamma"]
        image=TF.adjust_gamma(image,gamma)

    if params["scale_label"]:
        label=scale_label(label,params["target_size"])
        
    image=TF.to_tensor(image)
    return image, label
#######################################33
#Adujsting random seed is important to get similar result from two code.
np.random.seed(0)
random.seed(0)
trans_params_train = {
        
        "target_size": (256,256),
        "p_hflip": 0.5,
        "p_vflip": 0.5,
        "p_shift": 0.5,
        "max_translate": (0.2,0.2),
        "p_brightness": 0.5,
        "brightness_factor": 0.2,
        "p_contrast": 0.2,
        "contrast_factor": 0.2,
        "p_gamma": 0.5,
        "gamma": 0.2,
        "scale_label": True,
        }
trans_params_val = {
        
        "target_size": (256,256),
        "p_hflip": 0,
        "p_vflip": 0,
        "p_shift": 0,
        "p_brightness": 0,
        "brightness_factor": 0,
        "p_contrast": 0,
        "p_gamma": 0,
        "gamma": 0,
        "scale_label": True,  
        }

amd_ds1=AMD_dataset(path2data,transformer,trans_params_train)
amd_ds2=AMD_dataset(path2data,transformer,trans_params_val)

#===================================================
#split the dataset into training and validation sets(output is index)

from sklearn.model_selection import ShuffleSplit
#"ShuffleSplit",contrary to other cross-validation strategies, 
#random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets.
#This means that "ShuffleSplit" randomly split a dataset into overlapping new datasets, otherwise dataset is sizeable
sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = range(len(amd_ds1))

for train_index, val_index in sss.split(indices):
    
    print(len(train_index))
    print('-'*10)
    print(len(val_index))

#Define the training and validation datasets
from torch.utils.data import Subset

train_ds=Subset(amd_ds1,train_index)
print(len(train_ds))

val_ds=Subset(amd_ds2,val_index)
print(len(val_ds))
# ====================================================
import matplotlib.pyplot as plt
import numpy as np 


np.random.seed(0)

plt.figure(figsize=(5,5))
for img,label in train_ds:
    show(img,label)
    break

plt.figure(figsize=(5,5))
for img,label in val_ds:
    show(img,label)
    break
# =====================================================
# Data Loader

from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

for img_b, label_b in train_dl:
    print(img_b.shape, img_b.dtype)
    print(label_b)
    break

# ======================================================
import torch

for img_b , label_b in train_dl:
    
    print(img_b.shape, img_b.dtype)
    
    label_b = torch.stack(label_b,1)
    label_b = label_b.type(torch.float32)
    
    print(label_b.shape, label_b.dtype)
    break

for img_b , label_b in val_dl:
    
    print(img_b.shape, img_b.dtype)
    
    label_b = torch.stack(label_b,1)
    label_b = label_b.type(torch.float32)
    
    print(label_b.shape, label_b.dtype)
    break

    
# ======================================Creating the model

import torch.nn as nn
import torch.nn.functional as F

class Net (nn.Module):
    
    def __init__(self, params):
        super(Net,self).__init__()
    def forward(self,x):
        return x
    
def __init__(self, params):
    
    super(Net, self).__init__()
        
    C_in,H_in,W_in = params["input_shape"]
    init_f = params["initial_filters"]
    num_outputs = params["num_outputs"]
        
    self.conv1 = nn.Conv2d(C_in, init_f, 
                               kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(init_f + C_in, 2*init_f, 
                               kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(3*init_f + C_in, 4*init_f, 
                               kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(7*init_f + C_in, 8*init_f, 
                               kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(15*init_f + C_in, 16*init_f, 
                               kernel_size=3, padding=1)
    self.fc1 = nn.Linear(16*init_f, num_outputs)
        
def forward(self, x):
        
    identity = F.avg_pool2d(x,4,4)
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = torch.cat((x,identity), dim=1)
        
    identity = F.avg_pool2d(x,2,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = torch.cat((x,identity), dim=1)
        
    identity = F.avg_pool2d(x,2,2)
    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x,2,2)
    x = torch.cat((x,identity), dim=1)
        
    identity = F.avg_pool2d(x,2,2)
    x = F.relu(self.conv4(x))
    x = F.max_pool2d(x,2,2)
    x = torch.cat((x,identity), dim=1)
        
    x = F.relu(self.conv5(x))
    x = F.adaptive_max_pool2d(x,1)
    x =  x.reshape(x.size(0), -1)
        
    x = self.fc1(x)
        
    return x
# === Override the Net class functions
Net.__init__ = __init__
Net.forward = forward

# === Define an Object the Net class function 

params_model = {
                "input_shape": (3,256,256),
                "initial_filters":16,
                "num_outputs":2,
                }

model = Net(params_model)
# === Move the model to CUDA
if torch.cuda.is_available():
    
    device = torch.device("cuda")
    model = model.to(device)

print(model)
    
# =========Define the loss function, optimizer, and the IOU metrics

loss_func =  nn.SmoothL1Loss(reduction="sum")

# =============================================================================
# =======for test
# n,c = 8,2
# y = 0.5 * torch.ones(n, c, requires_grad = True)
# print(y.shape)
# 
# target = torch.zeros(n, c, requires_grad= False)
# print(target.shape)
# 
# loss = loss_func(y, target)
# print(loss.item())
# 
# y = 2*torch.ones(n,c,requires_grad=True)
# target = torch.zeros(n, c, requires_grad=False)
# loss = loss_func(y, target)
# print(loss.item())
# =============================================================================
from torch import optim
opt = optim.Adam(model.parameters(), lr =3e-4)

def get_lr(opt):
    
    for param_group in opt.param_groups:
        return param_group['lr']

# =============================================================================
# ===for test
# current_lr = get_lr(opt)
# print('current lr = {}'.format(current_lr))
# =============================================================================

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode= 'min', factor = 0.5 , 
                                 patience=20,verbose=1)

# =============================================================================
# ====for test
# for i in range(100):
#     lr_scheduler.step(1)
# =============================================================================
# ===== convert coordinates to a bounding box
def cxcy2bbox(cxcy, w=50./256, h=50./256):
    
    w_tensor = torch.ones(cxcy.shape[0],1,device=cxcy.device)*w
    h_tensor = torch.ones(cxcy.shape[0],1,device=cxcy.device)*h
    cx = cxcy[:,0].unsqueeze(1)
    cy = cxcy[:,1].unsqueeze(1)
    
    boxes = torch.cat((cx,cy,w_tensor,h_tensor),-1)
    return torch.cat((boxes[:, :2]-boxes[:,2:]/2
                      ,boxes[:, :2]+boxes[:,2:]/2),1)


torch.manual_seed(0)
cxcy = torch.rand(1,2)
print("center:",cxcy*256)

bb = cxcy2bbox(cxcy)
print("bounding box", bb*256)

# ===== Define the metric

import torchvision
def metrics_batch (output, target):
    
    output = cxcy2bbox(output)
    target = cxcy2bbox(target)
    iou = torchvision.ops.box_iou(output, target)
    return torch.diagonal(iou, 0).sum().item()
#==== Try it out on known values
# =============================================================================
# n,c = 8,2
# target = torch.rand(n, c, device=device)
# target = cxcy2bbox(target)
# metrics_batch(target, target)
# ============================================================================= 
# ==== Loss_batch function
def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b
# ====Try the loss_batch function(Unit test)
# =============================================================================
# for xb, label_b in train_dl:
#     
#     label_b = torch.stack(label_b,1)
#     label_b = label_b.type(torch.float32)
#     label_b = label_b.to(device)
#     
#     l,m = loss_batch(loss_func, label_b, label_b)
#     print(l,m)
#     break
# =============================================================================
# ==== Train and Evaluation
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    
    for xb, yb in dataset_dl:
        yb = torch.stack(yb,1)
        yb = yb.type(torch.float32).to(device)
        output = model(xb.to(device))
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
            
        if sanity_check is True:
            break
    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    return loss,metric

import copy
def train_val(model, params):
    
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history = {
        "train":[],
        "val": [], 
        } 
    metric_history={
        "train": [],
        "val": [],
        }  
     
    best_model_wts = copy.deepcopy(model.state_dict)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch{}/{}, current lr={}'.format
              (epoch, num_epochs-1, current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(
            model, loss_func, train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),path2weights)
            print("Copied best model weights!")
        
        lr_scheduler.step(val_loss)
        if current_lr !=get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        print("train loss: %.6f, accuracy: %.2f"
              %(train_loss,100*train_metric))
        print("val loss: %.6f, accuracy:%.2f"
              %(val_loss,100*val_metric))
        print("_"*10)
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history
# === Train the model by calling train_val function
loss_func = nn.SmoothL1Loss(reduction="sum")
opt = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5
                                 ,patience=20,verbose=1)
path2models="./models/"
if not os.path.exists(path2models):
    os.mkdir(path2models)

params_train={
    "num_epochs": 10,
    "optimizer":opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2models+"weights_smoothl1.pt"
    }
model, loss_hist, metric_hist = train_val(model, params_train)

# ==== Plot the training and validation loss
num_epochs = params_train["num_epochs"]
plt.title("Train-val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1), metric_hist["train"], label="train")
plt.plot(range(1,num_epochs+1), metric_hist["val"], label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()






    
    




    






   
   
   
    


    
    
    



