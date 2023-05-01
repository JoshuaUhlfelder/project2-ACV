# Detecting Ink from Ancient Roman Scrolls by Training Deep Learning Models
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

# Model 1: Mask RCNN
1. RCNN.py - the original Mask RCNN trainer that validates on an entire valdiation fragment to get the F0.5 score
2. RCNN-with-val.py - trains a Mask RCNN using validation windows (not entire fragments) from a fragment
3. RCNN-with-val-thin - trains a Mask RCNN using only 6 layers of training data (not 65)
4. RCNN-test - reassmbles a test fragment using a completed model by predicting the output on windows across the fragment and piecing them together

# Model 2: Transformer
1. Transformer-with-val.py - trains a transformer model using validation windows from a fragment
2. trasformer-test.py - reassmbles a test fragment using a completed model by predicting the output on windows across the fragment and piecing them together

# Model 3: U-Net Variation
1. UNET-with-val.py - trains a 65-channeled U-Net using validation windows from a fragment
2. UNET-test.py - reassmbles a test fragment using a completed model by predicting the output on windows across the fragment and piecing them together

# Model 4: Final U-Net
1. model_4_requirements.txt - a requirements file to run our final and best model: the U-Net
2. model_4_unet_train-review.ipynb - the training and validation file for the model

# Code Reviews and Team Responsibilities
1. Code Reviews - a folder containing old project files with the other team member's comments
2. Team Responsibilities.md - describes the specific tasks of each team member
