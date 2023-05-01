
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
import transformers
from transformers import (
    AutoImageProcessor,
    ViTModel,
    )


"""
TRANSFORMER TEST

CODE REVIEW
Code by Josh
Reviewed by Joe

NOTE: CODE REVIEW COMMENTS USE QUOTES ("")
-Traditional code comments use #
"""

"""
Revise comments below. They seem old
The max image size is not set here
"""
#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
test_dir = "../vesuvius-challenge-ink-detection/test"
#A new 'masks' folder will be created in this directory


"""
You could potentially print out the device for a better user experience
"""
# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Load in model and make the same changes as before.

model = ViTModel.from_pretrained(
    "google/vit-base-patch16-224",
    ignore_mismatched_sizes=True,
)
model.embeddings.patch_embeddings.projection = nn.Conv2d(65, 768, kernel_size=(16, 16), stride=(16, 16))

#Custom layer to convert 1d input to 2d
class MultiDimLinear(torch.nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)
    """
    Ensure this works for multi-batch data.
    By reshaping the output to (1,32,32),
    where do the other items in the batch go?
    Shouldn't it reshape to (1,1,32,32)
    """
    def forward(self, x):
        out = super().forward(x)
        return out.reshape((1, 32, 32))



model.pooler.dense = nn.Sequential(
    nn.Linear(in_features=768, out_features=1024, bias=True),
    MultiDimLinear(1024,(32,32)),
    nn.ConvTranspose2d(1,1,2,stride=2),
    nn.ConvTranspose2d(1,1,2,stride=2),
    nn.ConvTranspose2d(1,1,2,stride=2),
    nn.Conv2d(1, 1, kernel_size=(18, 18),padding=0),
    nn.Conv2d(1, 1, kernel_size=(16, 16),padding=0),
)

"""
You already have this above. Either delete it here or above.
"""
#Load the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Load states
model.load_state_dict(torch.load('../model3_final', map_location=torch.device(device)))
model.to(device)# move model to the right devic
model.eval()


img_collections=[]
for pth in os.listdir(test_dir):
    img_collections.append(test_dir+ '/' + pth + '/')
    
#img_collections=[img_collections[0]]

"""
You can delete the img_collections loop above if you are
going to set it to a set variable.
"""
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
    #true_mask = cv2.resize(true_mask, [224,224], cv2.INTER_NEAREST)
    #Ensure the selected center of window is within the fragment
    big_mask.close()
    """
    You can close big_mask earlier to save space (line 119)
    """
    
    combined = np.zeros((height, width))
    
    """
    For loop is unecessary. you are only iterating over one item.
    But you can keep this is you want the flexibility to add more
    """
    for l in [640]:
        """
        You do not need to set max_image_size to l. Instead, you could
        just rename l to max_image_size.
        """
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
                images = np.zeros((224,224,65))
                layer_names = os.listdir(img_collections[i] + 'surface_volume')
                layer_names.sort()
                #Add each layer of the fragment to the stack of images
                for j in range(65):
                    with rasterio.open(img_collections[i] + 'surface_volume/' + layer_names[j]) as img:
                        chunk = img.read(1, window=window)
                        chunk = cv2.resize(chunk, [224,224], cv2.INTER_LINEAR)
                        images[:,:,j] = chunk
                #Add the chunk to the image file
                #Track the coordinates of the image so we can reassemble the large picture
    
                inputs = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
                inputs=inputs.swapaxes(1, 3).swapaxes(2, 3)
                inputs = inputs.to(device)
        
                with torch.no_grad():
                    outputs = model(inputs.float())
                    """
                    The items below do not need to be within the with statement
                    """
                    preds = outputs.pooler_output[0,:,:].detach().cpu().numpy()
                    preds = cv2.resize(preds, [max_image_size,max_image_size], cv2.INTER_LINEAR)
                    
                    preds = np.where(preds>0.5, 1, 0)
                    #print(preds.max())
                    
                result[coords[0]:(coords[0]+max_image_size), 
                       coords[1]:(coords[1]+max_image_size)] = preds
                
        combined = np.add(combined, result)
        
    final = np.where(combined>0, 1, 0)
    cv2.imwrite('../' + str(i) + '4.png', final*255)
    
    #Get the f1 score
    def score(true_mask, predicted):
        """
        The TN variable is uncessary. You never use it in the calculations below
        """
        TP = np.sum(np.logical_and(predicted == 1, true_mask == 1))
        TN = np.sum(np.logical_and(predicted == 0, true_mask == 0))
        FP = np.sum(np.logical_and(predicted == 1, true_mask == 0))
        FN = np.sum(np.logical_and(predicted == 0, true_mask == 1))
        
        print('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
        
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        B = 0.5
        
        """
        Also, it might be helpful to print out p and r to gives us more metrics
        (precision, recall)
        """
        
        #Calculate the F0.5 score
        scr5 = ((1+math.pow(B, 2))*p*r)/(math.pow(B, 2) * p + r)
        return scr5
    
    
    
    f_5_score = score(true_mask, final)
    print('F0.5 score:', f_5_score)
    
    
