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

"""
There is no variable to define the image size like you claimed in your comment
Could this comment be left over from copying the validation code?
Or are you missing this variable?
"""
#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
#The maximum window size will be max_image_size x max_image_size

"""
Your comment here says there is a mask directory. Where? This does not seem incldued.
"""
#Set training and mask directory
test_dir = "../vesuvius-challenge-ink-detection/test"
#A new 'masks' folder will be created in this directory

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
"""
Suggestion: you could print the device to validate for the user how the code
is being run
"""



#Load in model and make the same changes as before.
model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, 
                                                         image_mean = torch.tensor(np.full(6,0.5)),
                                                         image_std = torch.tensor(np.full(6,0.25)))
model.backbone.body.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features 
#Set it to only two classes - ink, no ink
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes=2)

#Load states
model.load_state_dict(torch.load("../checkpoints/9_400.torch", map_location=torch.device(device)))
model.to(device)# move model to the right devic
model.eval()


img_collections=[]
for pth in os.listdir(test_dir):
    img_collections.append(test_dir+ '/' + pth + '/')
    
#img_collections=[img_collections[0]]

img_collections = ['../vesuvius-challenge-ink-detection/train/1/']
    
for i in range(len(img_collections)):
        
    #Create the big mask with the same dimensions as the og. files
    big_mask = rasterio.open(img_collections[i] + 'inklabels.png')
    #big_mask = rasterio.open(img_collections[i] + 'mask.png')
    height = big_mask.height
    width = big_mask.width
    window = Window(0,0,width,height)
    true_mask = big_mask.read(1, window=window)
    true_mask = (true_mask > 0).astype(np.uint8)
    #Ensure the selected center of window is within the fragment
    big_mask.close()
    
    """
    The mask file does not need to be open past line 80. You could
    close it earlier to save space.
    """
    
    combined = np.zeros((height, width))
    
    for l in [600,510,430,380]:
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
                """
                Will print many times. Sugegstion: only print certain cycles to
                ensure the program is operating properly without spamming
                the console.
                """
                print(window)
                images = np.zeros((max_image_size,max_image_size,6))
                layer_names = os.listdir(img_collections[i] + 'surface_volume')
                layer_names.sort()
                #Add each layer of the fragment to the stack of images
                """
                In the below section, the j+28 seems arbitrary. I see you are trying
                to get the middle 6 layers. Instead, you instantiate a new variable
                that finds the middle and then adds it to j.
                This could make your code more user-friendly.
                """
                for j in range(6):
                    with rasterio.open(img_collections[i] + 'surface_volume/' + layer_names[j+28]) as img:
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
                """
                Below, im does not need 3 channels... right? Instead, you could just
                make it a 1-channel matrix to save space. It's also a pretty big file,
                so doing so will make the program run faster.
                """
                im = np.zeros((max_image_size,max_image_size,3))
                for k in range(len(pred[0]['masks'])):
                    msk=pred[0]['masks'][k,0].detach().cpu().numpy()
                    scr=pred[0]['scores'][k].detach().cpu().numpy()
                    if scr>0.6 :
                        im[:,:,0][msk>0.5] = 1
                        im[:, :, 1][msk > 0.5] = 1
                        im[:, :, 2][msk > 0.5] = 1
                """
                If you take the suggestion above, make sure you change this to 2D
                """
                result[coords[0]:(coords[0]+max_image_size), 
                       coords[1]:(coords[1]+max_image_size), :] = im
                
                
        result = result[:,:,0]
        combined = np.add(combined, result)
    """
    For the voting threshold below, I suggest you make the 2 a flexible variable
    at the start of the code so that the user has more flexibility in adjusting
    how many votes it takes to make the mask positive.
    """
    final = np.where(combined>2, 1, 0)
    cv2.imwrite('../' + str(i) + '5.png', final*255)
    
    #Get the f1 score
    def score(true_mask, predicted):
        
        """
        TN is never used in your calculations. You could get rid of this variable.
        """
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
    
    
    
    f_5_score = score(true_mask, final)
    print('F0.5 score:', f_5_score)
    
    
