# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:17:59 2021

@author: asus
"""
params_model = {
    
    "input_shape":(3,256,256),
    "initial_filters":16,
    "num_outputs":2, 
    }

model = Net(params_model)
model.eval()

if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)

path2weights = "./models/weights.pt"
model.load_state_dict(torch.load(path2weights))
loss_func = nn.SmoothL1Loss(reduction="sum")

with torch.no_grad():
    loss,metric=loss_epoch(model,loss_func,val_dl)
print(loss,metric)
# ==========
from PIL import ImageDraw
import numpy as np
import torchvision.transforms.functional as tv_F

np.random.seed(0)
import matplotlib.pylab as plt

def show_tensor_2labels(img,label1,label2,w_h=(50,50)):
    
    label1 = rescale_label(label1,img.shape[1:])
    label2 = rescale_label(label2,img.shape[1:])
    img=tv_F.to_pil_image(img)
    
    w,h = w_h
    cx, cy=label1
    draw = ImageDraw.Draw(img)
    draw.rectangle(((cx-w/2, cy-h/2),(cx+w/2,cy+h/2)),outline="green", width=2)
    
    cx,cy = label2
    
    draw.rectangle(((cx-w/2, cy-h/2),(cx+w/2, cy+h/2)),outline="red", width=2)
    plt.imshow(np.asarray(img))
    
    rndInds = np.random.randint(len(val_ds),size=10)
    print(rndInds)
    
    plt.rcParams['figure.figsize'] = (15,10)
    plt.subplots_adjust(wspace=0.0, hspace=0.15)
    for i, rndi in enumerate(rndInds):
        img,label=val_ds[rndi]
        h,w = img.shape[1:]
        with torch.no_grad():
            label_pred = model(img.unsqueeze(0).to(device))[0].cpu()
        plt.subplot(2,3,i+1)
        show_tensor_2labels(img, label, label_pred)
        # ===Calculate IOU
        label_bb = cxcy2bbox(label_bb, label_pred_bb)
        label_pred_bb = cxcy2bbox(label_pred.unsqueeze(0))
        iou = torchvision.ops.box_iou(label_bb, label_pred_bb)  
        plt.title("%.2f" %iou.item())
        
        if i>4:
            break  
        
path2labels = os.path.join(path2data, "Training400", "Fovea_location.xlsx")
labels_df = pd.read_excel(path2labels,index_col="ID")
    
img, label= resize_img_label(img, label, target_size=(256,256))
print(img.size, label)
    
img = TF.to_tensor(img)
label = scale_label(label,(256,256))
print(img.shape)
    
with torch.no_grad(): 
   label_pred = model(img.unsqueeze(0).to(device))[0].cpu()
show_tensor_2labels(img, label, label_pred)

# ==== Calculate the inference time per image.

import time
elapsed_times = []
with torch.no_grad():
    for k in range(100):
        start = time.time()
        label_pred = model(img.unsqueeze(0).to(device))[0].cpu()
        elapsed = time.time() - start
        elapsed_times.append(elapsed)
print("infrence time per image: %.4f s" %np.mean(elapsed_times))
    
    
        
