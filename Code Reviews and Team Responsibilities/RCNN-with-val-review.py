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

CODE REVIEW
Code by Josh
Reviewed by Joe

NOTE: CODE REVIEW COMMENTS USE QUOTES ("")
-Traditional code comments use #
"""


#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
max_image_size= 224 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
train_dir="../vesuvius-challenge-ink-detection/train"
#A new 'masks' folder will be created in this directory
"""
The comment above is not relevant. There is no 'mask' folder
"""



#Get all surface_volume folders from the training directory
#Make the masks for each fragment
img_collections=[]
for pth in os.listdir(train_dir):
    if not pth.startswith('.'):
        img_collections.append(train_dir+ '/' + pth + '/')
img_collections.sort()
"""
By sorting the images, it assumes that the folder contents will always be
the same. A more robust approach might be to extract the specific names 
from the list instead of indecies while loading in data.
"""

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
        window_center_width = 0
        window_center_height = 0
        #center_val = 0
        
        
        window_center_width = random.randint(window_size,width - window_size - 1)
        window_center_height = random.randint(window_size,height - window_size - 1)
          
                
        """
        You can close big_mask earlier (line 76) to save space
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
            """
            It seems the resizing step is not necessary, as the mask should already
            by the max_image_size. 
            """
            vesMask=cv2.resize(vesMask,[max_image_size,max_image_size],cv2.INTER_NEAREST)
            #Find the connected regions
            (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(vesMask, connectivity=8)
            #Create a new mask for each connected region
            for i in range(1, numLabels):
                new_mask = np.where(labels == i, 1, 0)
                """
                You set the locations in the matrix equal to 1 where the label matches i,
                so you should not need to change the type of new_mask to uint8 below.
                """
                new_mask = (new_mask > 0).astype(np.uint8) 
                masks.append(new_mask)
        """
        You could add comments below to better explain the functionality. I'm
        having trouble understanding the aggregation of the batch
        """
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

 



"""
This function seems to be the same as the below with minor modifications
Assume all comments for loadData() apply to this function as well.
"""

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
        """
        If you're going to essentially throw away the large image tensor
        you already made by recalling this function, maybe move the
        mask creation function above the image creation, so that
        you don't waste time.
        """
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











"""
Add comments about the specific chanegs you are making the the model below
"""
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
    
    """
    You never use TN below. You could take this variable out.
    """
    TP = np.sum(np.logical_and(predicted == 1, true_mask == 1))
    TN = np.sum(np.logical_and(predicted == 0, true_mask == 0))
    FP = np.sum(np.logical_and(predicted == 1, true_mask == 0))
    FN = np.sum(np.logical_and(predicted == 0, true_mask == 1))
    
    #print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    
    p = TP/(TP+FP)
    r = TP/(TP+FN)
    B = 0.5
    
    #Calculate the F0.5 score
    scr5 = ((1+math.pow(B, 2))*p*r)/(math.pow(B, 2) * p + r)
    return scr5

"""
This seciton is a bit light on comments. It's hard to understand what is going on
"""
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
        """
        Setting m=0 assumes that the first image in img_collections will always
        be your validation fragment. You could make this more robust by
        calling the spcific name of the fragment.
        """
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
                """
                It seems you might not need to resize the mask. 
                it should already be the correct dimensions.
                """
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
        
        
        
        
        
        
        
        
        

           

