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
trains a Mask RCNN using validation windows (not entire fragments) from a fragment

The data loading process and training loop are adapted from Sagi Eppel:
    https://towardsdatascience.com/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3
The mask-loading process and final image-stitching algorithms are original.
"""


#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
max_image_size= 224 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
train_dir="../vesuvius-challenge-ink-detection/train"
#A new 'masks' folder will be created in this directory


#Get all surface_volume folders from the training directory
#Make the masks for each fragment
img_collections=[]
for pth in os.listdir(train_dir):
    if not pth.startswith('.'):
        img_collections.append(train_dir+ '/' + pth + '/')
img_collections.sort()

#Tablet 1 used for testing. Tablets 2 and 3 used for training.


#Load in data
def loadData():
    batch_Imgs=[]
    batch_Data=[]
    for i in range(batch_size):
        #Randomly select a target image file from the training tablets
        idx=random.randint(1,len(img_collections)-1)
        #Get the names of all the tifs in the selected folder
        layer_names = os.listdir(img_collections[idx] + 'surface_volume')
        
        #Get a random window size (radius) with the maximum dimensions as image_size
        #And a min size as 20% of that size
        window_size = math.floor(random.randint(math.floor(0.3*max_image_size),max_image_size*3)/2)
        
        #Get the center of the window on the image
        big_mask = rasterio.open(img_collections[idx] + 'mask.png')
        height = big_mask.height
        width = big_mask.width
        big_mask.close()
        window_center_width = 0
        window_center_height = 0
        #center_val = 0
        
        window_center_width = random.randint(window_size,width - window_size - 1)
        window_center_height = random.randint(window_size,height - window_size - 1)
                
        
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
        masks = np.array(masks)
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



def loadValData():
    batch_Imgs=[]
    batch_Data=[]
    window_Data=[]
    #First tablet is validation data
    idx=0
    #Get the names of all the tifs in the selected folder
    layer_names = os.listdir(img_collections[idx] + 'surface_volume')
    
    #Validation data is always max_image_size x max_image_size
    window_size = math.floor(max_image_size/2)
    
    #Get the center of the window on the image
    big_mask = rasterio.open(img_collections[idx] + 'mask.png')
    height = big_mask.height
    width = big_mask.width
    window_center_width = 0
    window_center_height = 0
    
    window_center_width = random.randint(window_size,width - window_size - 1)
    window_center_height = random.randint(window_size,height - window_size - 1)

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
        for i in range(1, numLabels):
            new_mask = np.where(labels == i, 1, 0)
            new_mask = (new_mask > 0).astype(np.uint8) 
            masks.append(new_mask)
        
        num_objs = len(masks)
        if num_objs==0: 
            return loadValData()

        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        full_img = torch.as_tensor(full_img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   
        data["masks"] = masks
        batch_Imgs.append(full_img)
        batch_Data.append(data)  
        window_Data.append(window)

    batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    print(window_Data)
    return [batch_Imgs, batch_Data, window_Data]



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
print(device)

#Set optimizer and lr
lr = 3e-4
print("Set learning rate to:", lr)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()




#Get the true/false positive/negatives for the created mask vs. the true mask
#Then output the F0.5 score
def score(true_mask, predicted):
    
    unique, counts = np.unique(predicted, return_counts=True)
    print("predicted counts:", dict(zip(unique, counts)))
    
    unique, counts = np.unique(true_mask, return_counts=True)
    print("true counts:", dict(zip(unique, counts)))
    
    TP = np.sum(np.logical_and(predicted == 1, true_mask == 1))
    FP = np.sum(np.logical_and(predicted == 1, true_mask == 0))
    FN = np.sum(np.logical_and(predicted == 0, true_mask == 1))
    
    #print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    
    p = TP/(TP+FP)
    r = TP/(TP+FN)
    B = 0.5
    
    #Calculate the F0.5 score
    scr5 = ((1+math.pow(B, 2))*p*r)/(math.pow(B, 2) * p + r)
    return scr5


#Training and evaluation loop
for i in range(401):
    model.train()
    images, targets = loadData()
    images = list(image.to(device) for image in images)
    targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
    
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    
    losses.backward()
    optimizer.step()
    
    print(i,'loss:', losses.item())
    #Evaluation
    if i%100==0:
        torch.save(model.state_dict(), '../checkpoints/' + str(i)+".torch")
        print("Save model to:",'../checkpoints/' + str(i)+".torch")
        print('\nEvaluating...')
        
        model.eval()

        m = 0 #Only first fragement used for validation

        print("testing 10 samples on fragment", img_collections[m])

        val_loops = 20
        
        all_losses = []
        f_5_scores = []
        for l in range(val_loops):
            
            #create an empty matrix for our mask we'll make
            print("Validation", l)

            [images, targets, windows] = loadValData()
            window = windows[0]
            images = list(image.to(device) for image in images)
            targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
            
            
            with torch.no_grad():
                pred = model(images)
                model.train()
                loss_dict = model(images, targets)
            this_loss = sum(loss for loss in loss_dict.values())
            model.eval()
                

                
            #Get the true mask
            mask_file=os.path.join(img_collections[m], 'inklabels.png')
            true_mask = None
            with rasterio.open(mask_file) as mk:
                true_mask = mk.read(1, window=window)
                true_mask = (true_mask > 0).astype(np.uint8) 
                #Resize the image
                true_mask=cv2.resize(true_mask,[max_image_size,max_image_size],cv2.INTER_NEAREST)
            
            im = np.zeros((max_image_size,max_image_size))
            for k in range(len(pred[0]['masks'])):
                msk=pred[0]['masks'][k,0].detach().cpu().numpy()
                scr=pred[0]['scores'][k].detach().cpu().numpy()
                if scr>0.6 :
                    im[:,:][msk>0.5] = 1
            
            f_5_scores.append(score(true_mask, im))
            all_losses.append(this_loss)
            
            
        print('Average F0.5 score:', sum(f_5_scores)/val_loops)
        print(f_5_scores)
        print('Average loss:', sum(all_losses)/val_loops)
        print(all_losses)
        
        
        
        
        
        
        
        
        

           

