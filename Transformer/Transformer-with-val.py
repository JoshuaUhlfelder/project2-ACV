

import transformers
from transformers import (
    AutoImageProcessor,
    ViTModel,
    )
import cv2
import os 
import numpy as np
import torchvision.models.segmentation
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import random
from torchvision import transforms

import rasterio
from rasterio.windows import Window
import math
import time
import copy
from tqdm import tqdm





#Define the batch size and the size every sample from the fragment will be converted to
batch_size=1
max_image_size= 640 #The maximum window size will be max_image_size x max_image_size

#Set training and mask directory
train_dir="../vesuvius-challenge-ink-detection/train"
#A new 'masks' folder will be created in this directory

#Set number of training windows for each fragment
#Each epoch will have train_cycle_count x fragment_count iterations
train_cycle_count = 50
val_cycle_count = 40


class MyTrainDataset(torch.utils.data.Dataset):
    """
    This class needs the directory where the fragment images are stored
    It also takes the list of indecies for training fragments
    """
    def __init__(
        #Needs directory of images, metadata file, and transformations
        self,
        images_dir,
        index_list
    ):
        # Collect images, info, and labels into lists
        self.fragment_folders = self.get_fragment_folders(
            images_dir,
            index_list
        )
        self.index_list = index_list
        # Convert to np arrays
        self.fragment_folders = np.array(self.fragment_folders)
        # Size of datasets
        self.num_fragments = len(self.fragment_folders)
        
    def __len__(self):
        return self.num_fragments
        
    #Only the folders used for training are extracted
    def get_fragment_folders(self, images_dir, index_list):

        img_collections=[]
        for pth in os.listdir(images_dir):
            if not pth.startswith('.'):
                img_collections.append(images_dir+ '/' + pth + '/')
        img_collections.sort()
        
        training_only = []
        for i in range(len(img_collections)):
            if i in index_list:
                training_only.append(img_collections[i])
                
        repeated_list = []
        for j in range(train_cycle_count):
            repeated_list += training_only
            
        return repeated_list

    
    def __getitem__(self, idx):  
        # Retrieve an image from the list, load it, transform it, 
        # and return it along with its label
        #Bad data returned as None
        #try:
        #Get the names of all the layers for the folder
        layer_names = os.listdir(self.fragment_folders[idx] + 'surface_volume')
        
        #Get a random window size (radius) with the maximum dimensions as image_size
        #And a min size as 10% of that size
        window_size = math.floor(random.randint(math.floor(0.2*max_image_size),max_image_size)/2)
        
        #Get the center of the window on the image
        big_mask = rasterio.open(self.fragment_folders[idx] + 'mask.png')
        height = big_mask.height
        width = big_mask.width
        big_mask.close()
        window_center_width = 0
        window_center_height = 0
        
        
        window_center_width = random.randint(window_size,width - window_size - 1)
        window_center_height = random.randint(window_size,height - window_size - 1)
                 
        #Set the window bounds
        window = Window(window_center_width - window_size, 
                        window_center_height - window_size, window_size*2, window_size*2)
        
        #Now get the ink labels for only the windowed region
        ink_file=os.path.join(self.fragment_folders[idx], 'inklabels.png')
        with rasterio.open(ink_file) as mk:
            ink_labels = mk.read(1, window=window)
        ink_labels = (ink_labels > 0).astype(np.uint8)
        #Resize the image
        ink_labels=cv2.resize(ink_labels,[224,224],cv2.INTER_NEAREST)
        #Make the 2d ink labels 3d
        ink_labels = ink_labels[:, :, np.newaxis]
        
        #Ensure that the training window has some ink in it
        if ink_labels.max()==0: 
            return self.__getitem__(idx)
        
        #for each TIF, open the file in the window and add to the tensor
        full_img = np.zeros((224,224,65))
        layer_names.sort()
        for j in range(65):
            with rasterio.open(self.fragment_folders[idx] + 'surface_volume/' + layer_names[j]) as img:
                chunk = img.read(1, window=window)
                chunk = cv2.resize(chunk, [224,224], cv2.INTER_LINEAR)
                full_img[:,:,j] = chunk


                    
        #Flip along horz. or vert. axis with prob 0.5
        if random.randint(0,1) == 0:
            full_img = np.flip(full_img, 0).copy()
            ink_labels = np.flip(ink_labels, 0).copy()
        if random.randint(0,1) == 0:
            full_img = np.flip(full_img, 1).copy()
            ink_labels = np.flip(ink_labels, 1).copy()
        
        full_img = torch.tensor(full_img)
        ink_labels = torch.tensor(ink_labels)
        
        return full_img, ink_labels
        #except Exception as exc:  # <--- i know this isn't the best exception handling
        #    return None



class MyValDataset(torch.utils.data.Dataset):
    """
    This class needs the directory where the fragment images are stored
    It also takes the list of indecies for validation fragments
    This class is different from the training dataset class because it 
    doesn't take a random window. Instead, it just uses the max_image_size
    """
    def __init__(
        #Needs directory of images, metadata file, and transformations
        self,
        images_dir,
        index_list
    ):
        # Collect images, info, and labels into lists
        self.fragment_folders = self.get_fragment_folders(
            images_dir,
            index_list
        )
        self.index_list = index_list
        # Convert to np arrays
        self.fragment_folders = np.array(self.fragment_folders)
        # Size of datasets
        self.num_fragments = len(self.fragment_folders)
        
    def __len__(self):
        return self.num_fragments
        
    #Only the folders used for training are extracted
    def get_fragment_folders(self, images_dir, index_list):

        img_collections=[]
        for pth in os.listdir(images_dir):
            if not pth.startswith('.'):
                img_collections.append(images_dir+ '/' + pth + '/')
        img_collections.sort()
        
        val_only = []
        for i in range(len(img_collections)):
            if i in index_list:
                val_only.append(img_collections[i])
        
        #Copy the fragments over by val_cycle_count times
        repeated_list = []
        for j in range(val_cycle_count):
            repeated_list += val_only
        print(len(repeated_list))
        return repeated_list

    
    def __getitem__(self, idx):  
        # Retrieve an image from the list, load it, transform it, 
        # and return it along with its label
        #Bad data returned as None
        try:
            #Get the names of all the layers for the folder
            layer_names = os.listdir(self.fragment_folders[idx] + 'surface_volume')
            
            #Set the window size to the max
            window_size = math.floor(max_image_size/2)
            
            #Get the center of the window on the image
            big_mask = rasterio.open(self.fragment_folders[idx] + 'mask.png')
            height = big_mask.height
            width = big_mask.width
            big_mask.close()
            window_center_width = 0
            window_center_height = 0
            
            
            window_center_width = random.randint(window_size,width - window_size - 1)
            window_center_height = random.randint(window_size,height - window_size - 1)
                     
            #Set the window bounds
            window = Window(window_center_width - window_size, 
                            window_center_height - window_size, window_size*2, window_size*2)
            
            #Now get the ink labels for only the windowed region
            ink_file=os.path.join(self.fragment_folders[idx], 'inklabels.png')
            with rasterio.open(ink_file) as mk:
                ink_labels = mk.read(1, window=window)
            ink_labels = (ink_labels > 0).astype(np.uint8) 
            #Resize the image
            ink_labels=cv2.resize(ink_labels,[224,224],cv2.INTER_NEAREST)
            #Make the 2d ink labels 3d
            ink_labels = ink_labels[:, :, np.newaxis]
            
            #Ensure that the validation window has some ink in it
            if ink_labels.max()==0: 
                return self.__getitem__(idx)
            
            #for each TIF, open the file in the window and add to the tensor
            full_img = np.zeros((224,224,65))
            layer_names.sort()
            for j in range(65):
                with rasterio.open(self.fragment_folders[idx] + 'surface_volume/' + layer_names[j]) as img:
                    chunk = img.read(1, window=window)
                    chunk = cv2.resize(chunk, [224,224], cv2.INTER_LINEAR)
                    full_img[:,:,j] = chunk
            
            
            
            
            full_img = torch.tensor(full_img)
            ink_labels = torch.tensor(ink_labels)
            
            return full_img, ink_labels
        except Exception as exc:  # <--- i know this isn't the best exception handling
            return None


print("Setting up datasets")
index_list = []
train_data = MyTrainDataset(train_dir, [1,2]) #use only fragment 2 and 3 for training
val_data = MyValDataset(train_dir, [0]) #use only fragment 1 for validation

dataset_sizes = {}
dataset_sizes['train'] = len(train_data)
dataset_sizes['val'] = len(val_data)




def collate_fn(batch):
    # Filter failed images first
    batch = list(filter(lambda x: x is not None, batch))
    
    # Now collate into mini-batches
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    
    #Rearrange to make channels item 2
    images = images.swapaxes(1, 3).swapaxes(2, 3)
    labels = labels.swapaxes(1, 3).swapaxes(2, 3)
    
    return images, labels


print("Setting up dataloaders")
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                             shuffle=True, num_workers=0, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                             shuffle=True, num_workers=0, collate_fn=collate_fn)
dataloaders = {}
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader



#MODEL CREATION


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

    def forward(self, x):
        out = super().forward(x)
        return out.reshape((1, 32, 32))


#Decoder at end 
model.pooler.dense = nn.Sequential(
    nn.Linear(in_features=768, out_features=1024, bias=True),
    MultiDimLinear(1024,(32,32)),
    nn.ConvTranspose2d(1,1,2,stride=2),
    nn.ConvTranspose2d(1,1,2,stride=2),
    nn.ConvTranspose2d(1,1,2,stride=2),
    nn.Conv2d(1, 1, kernel_size=(18, 18),padding=0),
    nn.Conv2d(1, 1, kernel_size=(16, 16),padding=0),
)




#Load the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#Get the true/false positive/negatives for the created mask vs. the true mask
#Then output the F0.5 score
def score(true_mask, predicted):
    
    unique, counts = np.unique(predicted, return_counts=True)
    #print("predicted counts:", dict(zip(unique, counts)))
    
    unique, counts = np.unique(true_mask, return_counts=True)
    #print("true counts:", dict(zip(unique, counts)))
    
    TP = np.sum(np.logical_and(predicted == 1, true_mask == 1))
    FP = np.sum(np.logical_and(predicted == 1, true_mask == 0))
    FN = np.sum(np.logical_and(predicted == 0, true_mask == 1))
    
    #print('TP: %i, FP: %i, FN: %i' % (TP,FP,FN))
    
    p = TP/(TP+FP)
    r = TP/(TP+FN)
    B = 0.5
    #print(p,r)
    
    #Calculate the F0.5 score
    scr5 = ((1+math.pow(B, 2))*p*r)/(math.pow(B, 2) * p + r)
    #print(scr5)
    if math.isnan(scr5):
        return 0
    else:
        return scr5


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        print(scheduler.get_last_lr())

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_score = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    
                    preds = outputs.pooler_output[0,:,:].detach().cpu().numpy()
                    #print(preds.max())
                    preds = np.where(preds>0.5, 1, 0)
                    
                    scr = score(labels[0,0,:,:].detach().cpu().numpy(), preds)
                    
                    
                    loss = criterion(outputs.pooler_output.reshape((1, 1, 224, 224)), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_score += scr
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_score / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} F0.5: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                if epoch_acc < best_acc:
                    print('Epoch F0.5 is not better. Skipping')
                else:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F0.5: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Load a pretrained model and reset final fully connected layer for this particular classification problem.



# Move the model to the correct device
model = model.to(device)
model = model.float()


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

# Using dice-loss - seems to be recommended for U-Net
criterion = DiceLoss()

# Setup the optimizer to update the model parameters
optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)


# Decay LR by a factor of 0.8 after a linear warmup
scheduler1 = lr_scheduler.LinearLR(optimizer_ft, start_factor=0.05, total_iters=3)
scheduler2 = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.7)
scheduler = lr_scheduler.SequentialLR(optimizer_ft, 
                                      schedulers=[scheduler1, scheduler2], milestones=[3])




# Train and evaluate.  
model = train_model(model, criterion, optimizer_ft, scheduler,
                       num_epochs=10)



torch.save(model.state_dict(), '../model32_final')



