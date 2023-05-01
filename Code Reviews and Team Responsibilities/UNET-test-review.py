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
UNET TEST

CODE REVIEW
Code by Josh
Reviewed by Joe

NOTE: CODE REVIEW COMMENTS USE QUOTES ("")
-Traditional code comments use #
"""

"""
The comments below do not seem updated
There is no max_image_size variable
"""
#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
test_dir = "../vesuvius-challenge-ink-detection/test"
#A new 'masks' folder will be created in this directory

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
"""
Suggestion: print the device to the console for better user experience
"""

#Load in model and make the same changes as before.
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
#Change first layer to have 65 channels
model.encoder1.enc1conv1 = nn.Conv2d(65, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

"""
You set the device above already. 
Either delete this one or the other
"""
#Load the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Load states
model.load_state_dict(torch.load('../model2_final', map_location=torch.device(device)))
model.to(device)# move model to the right devic
model.eval()

"""
There is no need to loop directory
You set the variable below to a set list.

You can dteele lines 69-74
"""
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
    You can close big_mask earlier to save space, as it
    is not used beyond line 84
    """
    
    combined = np.zeros((height, width))
    
    for l in [640,512,384]:
        """
        There is no need to set max_image_size to l. Instead, just
        loop with max_image_size instead of l.
        """
        max_image_size= l
        
        #Get the number of full blocks in the x and y directions
        #This helps us divide up the image
        x_range = math.ceil(width/max_image_size)
        y_range = math.ceil(height/max_image_size)
        
        #create an empty matrix for our mask we'll make
        result = np.zeros((height, width))
        """
        Good: result is only 2D, which is teh best implementaiton for
        the test files so far.
        """
        
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
                Warning: you will be print out window a lot. Make sure this is
                something you want. 
                Instead, you could just print it for the first to ensure
                it is working properly.
                """
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
    
                inputs = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
                inputs=inputs.swapaxes(1, 3).swapaxes(2, 3)
                inputs = inputs.to(device)
        
                with torch.no_grad():
                    outputs = model(inputs.float())
                    """
                    The below statements do not need to be within the with statement
                    """
                    preds = outputs[0,0,:,:].detach().cpu().numpy()
                    preds = np.where(preds>0.5, 1, 0)
                
                result[coords[0]:(coords[0]+max_image_size), 
                       coords[1]:(coords[1]+max_image_size)] = preds
                
        combined = np.add(combined, result)
    """
    The threshold requires 2 of the 3 images to have ink
    Suggestion: You could make this automatic
    by taking the length of the l vector minus 1
    This way, if you adjust the l list, this will update
    automatrically.
    """
    final = np.where(combined>2, 1, 0)
    cv2.imwrite('../' + str(i) + '4.png', final*255)
    
    #Get the f1 score
    def score(true_mask, predicted):
        
        TP = np.sum(np.logical_and(predicted == 1, true_mask == 1))
        """
        TN is never used below except for in the print statement
        You can remove this variable
        """
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
    
    
