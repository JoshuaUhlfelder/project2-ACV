# Using Pre-Trained Deep Learning Models to Identify Underlying Diseases from Skin Lesion Image Data
ACV-project2
Joshua Uhlfelder, Joe Pehlke
# 

README.md

The goal of this project is to identify the location of ink from ancient roman scroll x-ray data. This repo is an entry in Kaggle's Vesuvius Ink Detection Challenge found here: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data. Using three pre-labeled scroll fragments which we divided into a training and validation set, we applied a variety of deep-learning models to attempt to solve this task. 

In this repo, there are five folders. The first four contain models used to solve the task. The last contains administrative items like code reviews and team responsibilities.
1. Model 1: Mask-RCNN - This folder has the training and test files that leverage a Mask R-CNN to find the location and shape of ink.
2. Model 2: Transformer - This folder contains two sub-optimal transformer models. The latter (Model 2.1) produces a low accuracy, while the former (2.0) is not functional.
3. Model 3: U-Net with 65-layered data - This application of U-Net trains using a full stack of training data (65 layers). 
4. Model 4: Final U-Net - This optimal model uses a slice of the full stack of training data to find the location of ink in fragments.. 
5. Administrative - This contains our code reviews and 'roles and responsibilities' file.

Each of the model folders has a training and test file or a combination, along with other variations described below. 

# Model 1
1. model1.py - trains a ResNet50 pretrained from ImageNet to classify skin lesions
2. model2.py - trains a ViT pretrained from ImageNet to classify skin lesions
3. model3.py - trains a BERT with a pretrained ResNet50 image encoder and a custom information encoder with demographic data about the skin lesion
4. model4.py - trains a binary classification BERT with a pretrained ResNet50 image encoder and a custom information encoder
5. model5.py - trains a BERT with a pretrained ResNet50 image encoder and a BERT text encoder with demographic data about the skin lesion

# Model 2
1. model1.py - trains a ResNet50 pretrained from ImageNet to classify skin lesions
2. model2.py - trains a ViT pretrained from ImageNet to classify skin lesions

# Model 3
1. model1.py - trains a ResNet50 pretrained from ImageNet to classify skin lesions
2. model2.py - trains a ViT pretrained from ImageNet to classify skin lesions

# Model 4
1. model1.py - trains a ResNet50 pretrained from ImageNet to classify skin lesions
2. model2.py - trains a ViT pretrained from ImageNet to classify skin lesions

# Administrative
1. Code Reviews
2. Roles and Responsibilities
