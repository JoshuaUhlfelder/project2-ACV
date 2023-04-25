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
batch_size=4
max_image_size= 1200 #The maximum window size will be max_image_size x max_image_size

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
        window_size = math.floor(random.randint(0.05*max_image_size,max_image_size)/2)
        
        #Get the center of the window on the image
        big_mask = rasterio.open(img_collections[idx] + 'mask.png')
        height = big_mask.height
        width = big_mask.width
        window_center_width = 0
        window_center_height = 0
        #center_val = 0
        
        
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
base_learning_rate = 3e-4
lr = base_learning_rate * batch_size / 256
print("Set learning rate to:", lr)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)




#Get the true/false positive/negatives for the created mask vs. the true mask
#Then output the F0.5 score
def score(true_mask, predicted):
    
    TP = np.sum(np.logical_and(predicted == 1, true_mask == 1))
    TN = np.sum(np.logical_and(predicted == 0, true_mask == 0))
    FP = np.sum(np.logical_and(predicted == 1, true_mask == 0))
    FN = np.sum(np.logical_and(predicted == 0, true_mask == 1))
    
    print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    
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

        m = 0
        print("testing on fragment", img_collections[m])
        ####Evaulation loop
        #Create the big mask with the same dimensions as the og. files
        big_mask = rasterio.open(img_collections[m] + 'inklabels.png')
        height = big_mask.height
        width = big_mask.width
        window = Window(0,0,width,height)
        true_mask = big_mask.read(1, window=window)
        true_mask = (true_mask > 0).astype(np.uint8)
        #Ensure the selected center of window is within the fragment
        big_mask.close()
        
        combined = np.zeros((height, width))
        
        for l in [1200, 840]:
            max_image_size= l
            
            #Get the number of full blocks in the x and y directions
            #This helps us divide up the image
            x_range = math.ceil(width/max_image_size)
            y_range = math.ceil(height/max_image_size)
            
            #create an empty matrix for our mask we'll make
            result = np.zeros((height, width))
            
            #Divide the image into chunks
            for x in range(x_range):
                for y in range(y_range):
                    coords = [y*max_image_size,x*max_image_size]
                    #If the image goes beyond the bounds, reset the coords back
                    if coords[0] + max_image_size >= height:
                        coords[0] = height-max_image_size
                    if coords[1] + max_image_size >= width:
                        coords[1] = width-max_image_size
                        
                    #for each chunk, make a window and load in the image
                    window = Window(coords[1], coords[0], max_image_size, max_image_size)
                    print(window)
                    images = np.zeros((max_image_size,max_image_size,65))
                    layer_names = os.listdir(img_collections[m] + 'surface_volume')
                    layer_names.sort()
                    #Add each layer of the fragment to the stack of images
                    for j in range(65):
                        with rasterio.open(img_collections[m] + 'surface_volume/' + layer_names[j]) as img:
                            chunk = img.read(1, window=window)
                            chunk = cv2.resize(chunk, [max_image_size,max_image_size], cv2.INTER_LINEAR)
                            images[:,:,j] = chunk
                    #Add the chunk to the image file
                    #Track the coordinates of the image so we can reassemble the large picture
        
                    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
                    images=images.swapaxes(1, 3).swapaxes(2, 3)
                    images = list(image.to(device) for image in images)
            
                    with torch.no_grad():
                        pred = model(images)
                    
                    im = np.zeros((max_image_size,max_image_size))
                    for k in range(len(pred[0]['masks'])):
                        msk=pred[0]['masks'][k,0].detach().cpu().numpy()
                        scr=pred[0]['scores'][k].detach().cpu().numpy()
                        if scr>0.6 :
                            im[:,:][msk>0.5] = 1
                    result[coords[0]:(coords[0]+max_image_size), 
                           coords[1]:(coords[1]+max_image_size)] = im
                        
            combined = np.add(combined, result)
            
            
        final = np.where(combined>1, 1, 0)
        
        f_5_score = score(true_mask, final)
        print('F0.5 score:', f_5_score)
        
        
        
        
        
        
        

           

