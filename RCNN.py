import cv2
import os 
import numpy as np
import torchvision.models.segmentation
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import random

import rasterio
from rasterio.windows import Window
import math


"""
MASK RCNN

Advising from https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
"""


#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
max_image_size= 600 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
train_dir="../vesuvius-challenge-ink-detection/train"
#A new 'masks' folder will be created in this directory
mask_dir = '../'


"""
Reads in a binary mask image with the ink classes
and creates a folder of individual mask files for each ink group.
"""
def make_masks(image, save_dir):
    
    #Convert the make image to a one-channel, binary image.
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    #CFind the connected regions
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    #Make new directories to store the mask files in
    try:
        os.mkdir('../masks')
    except:
        print("Note: mask directory already present.")
    sub_dir = image.split('/')[-2]
    
    try:
        os.mkdir('../masks/' + sub_dir)
    except:
        print("Note: mask sub-directory already present.")
    
    
    #For each connected region, create and save a new mask file
    for i in range(1, numLabels):
        new_mask = np.where(labels == i, 255, 0)
        split_name = 'mask_' + str(i) + '.png'
        cv2.imwrite(save_dir + 'masks/' + sub_dir + '/' + split_name, new_mask)
        



#Get all surface_volume folders from the training directory
#Make the masks for each fragment
img_collections=[]
for pth in os.listdir(train_dir):
    #print("Making masks for fragment", pth)
    img_collections.append(train_dir+ '/' + pth + '/')
    #make_masks('../vesuvius-challenge-ink-detection/train/' + pth + '/inklabels.png', mask_dir)



#Load in data
def loadData():
    batch_Imgs=[]
    batch_Data=[]
    for i in range(batch_size):
        #Randomly select a target image file
        idx=random.randint(0,len(img_collections)-1)
        #Get the names of all the tifs in the selected folder
        layer_names = os.listdir(img_collections[idx] + 'surface_volume')
        
        #Get a random window size (radius) with the maximum dimensions as image_size
        #And a min size as 20% of that size
        window_size = math.floor(random.randint(0.2*max_image_size,max_image_size)/2)
        
        #Get the center of the window on the image
        big_mask = rasterio.open(img_collections[idx] + 'mask.png')
        height = big_mask.height
        width = big_mask.width
        window_center_width = 0
        window_center_height = 0
        center_val = 0
        
        
        window_center_width = random.randint(window_size,width - window_size - 1)
        window_center_height = random.randint(window_size,height - window_size - 1)
        
        """
        #Ensure the selected center of window is within the fragment
        while(center_val != 1):
            window_center_width = random.randint(window_size,width - window_size - 1)
            window_center_height = random.randint(window_size,height - window_size - 1)
            for val in big_mask.sample([(window_center_width, window_center_height)]): 
                center_val = val[0]
       """     
                
                
        big_mask.close()
        #Set the window bounds
        window = Window(window_center_width - window_size, 
                        window_center_height - window_size, window_size*2, window_size*2)

        #for each TIF, open the file in the window and add to the tensor
        full_img = np.zeros((max_image_size,max_image_size,65))
        layer_names.sort()
        for j in range(65):
            with rasterio.open(img_collections[idx] + 'surface_volume/' + layer_names[j]) as img:
                chunk = img.read(1, window=window)
                chunk = cv2.resize(chunk, [max_image_size,max_image_size], cv2.INTER_LINEAR)
                full_img[:,:,j] = chunk

        #Now window and resize the mask and create a new mask for each glob
        masks=[]
        mask_file=os.path.join(img_collections[idx], 'inklabels.png')
        with rasterio.open(mask_file) as mk:
            vesMask = mk.read(1, window=window)
            vesMask = (vesMask > 0).astype(np.uint8) 
            #Resize the image
            vesMask=cv2.resize(vesMask,[max_image_size,max_image_size],cv2.INTER_NEAREST)
            #Find the connected regions
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(vesMask, connectivity=8)
            #Create a new mask for each connected region
            print(numLabels)
            for i in range(1, numLabels):
                new_mask = np.where(labels == i, 1, 0)
                new_mask = (new_mask > 0).astype(np.uint8) 
                masks.append(new_mask)
        
        num_objs = len(masks)
        if num_objs==0: 
            return loadData()

        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        full_img = torch.as_tensor(full_img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   
        data["masks"] = masks
        batch_Imgs.append(full_img)
        batch_Data.append(data)  
  
    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


#Load in the Mask RCNN model
model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, 
                                                         image_mean = torch.tensor(np.full(65,0.5)),
                                                         image_std = torch.tensor(np.full(65,0.25)))
model.backbone.body.conv1 = nn.Conv2d(65, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


in_features = model.roi_heads.box_predictor.cls_score.in_features 

#Set it to only two classes - ink, no ink
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)

#Load the model to the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#Set optimizer and lr
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)

model.train()


for i in range(10001):
   images, targets = loadData()
   images = list(image.to(device) for image in images)
   targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
   
   optimizer.zero_grad()
   loss_dict = model(images, targets)
   losses = sum(loss for loss in loss_dict.values())
   
   losses.backward()
   optimizer.step()
   
   print(i,'loss:', losses.item())
   if i%201==0:
           torch.save(model.state_dict(), '../' + str(i)+".torch")
           print("Save model to:",'../' + str(i)+".torch")

