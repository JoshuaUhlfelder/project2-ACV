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
 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
test_dir = "../vesuvius-challenge-ink-detection/train"
#A new 'masks' folder will be created in this directory

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Load in model and make the same changes as before.
model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, 
                                                         image_mean = torch.tensor(np.full(65,0.5)),
                                                         image_std = torch.tensor(np.full(65,0.25)))
model.backbone.body.conv1 = nn.Conv2d(65, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features 
#Set it to only two classes - ink, no ink
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)

#Load states
model.load_state_dict(torch.load("200.torch"))
model.to(device)# move model to the right devic
model.eval()


img_collections=[]
for pth in os.listdir(test_dir):
    img_collections.append(test_dir+ '/' + pth + '/')

img_collections = ['../vesuvius-challenge-ink-detection/train/1/']
    
for i in range(len(img_collections)):
        
    #Create the big mask with the same dimensions as the og. files
    big_mask = rasterio.open(img_collections[i] + 'mask.png')
    height = big_mask.height
    width = big_mask.width
    #Ensure the selected center of window is within the fragment
    big_mask.close()
    
    combined = np.zeros((height, width, 3))
    
    for l in [1200,900,600,300]:
        max_image_size= l
        
        #Get the number of full blocks in the x and y directions
        #This helps us divide up the image
        x_range = math.ceil(width/max_image_size)
        y_range = math.ceil(height/max_image_size)
        
        #create an empty matrix for our mask we'll make
        result = np.zeros((height, width, 3))
        
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
                layer_names = os.listdir(img_collections[i] + 'surface_volume')
                layer_names.sort()
                #Add each layer of the fragment to the stack of images
                for j in range(65):
                    with rasterio.open(img_collections[i] + 'surface_volume/' + layer_names[j]) as img:
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
                
                im = np.zeros((max_image_size,max_image_size,3))
                for k in range(len(pred[0]['masks'])):
                    msk=pred[0]['masks'][k,0].detach().cpu().numpy()
                    scr=pred[0]['scores'][k].detach().cpu().numpy()
                    if scr>0.6 :
                        im[:,:,0][msk>0.5] = 1
                        im[:, :, 1][msk > 0.5] = 1
                        im[:, :, 2][msk > 0.5] = 1
                result[coords[0]:(coords[0]+max_image_size), 
                       coords[1]:(coords[1]+max_image_size), :] = im
                    
        combined = np.add(combined, result)
        
    final = np.where(combined>2, 255, 0)
    cv2.imwrite('../' + str(i) + '.png', final)
    
    
