# Fish-Image-Classification-Streamlit

This project involves classifying a fish images dataset into multiple categories using deep learning models. It involves training a CNN model from scratch and leveraging transfer learning with pre-trained models like VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0 to enhance performance.It also involves comparing the accuracies of different models and saving the model with highest accuracy.
It also includes creating and deploying a Streamlit application to predict the fish categories from user-uploaded images based on the high accuracy model saved.

## Installation

To install the project the following packages are needed to be imported(if not already present):

- import pandas as pd
- import numpy as np
- import os
- import matplotlib.pyplot as plt
- import tensorflow as tf
- from pathlib import Path
- from keras.optimizers import Adam
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
- from tensorflow.keras import regularizers, layers, models
- from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 

To install packages run command on Windows: 
python -m pip install <Package Name>

# Execution / Usage

## Models
The following steps are to be performed for each model:
### 1. Get images and corresponding labels
### 2. Split the Fish image data into train and test
### 3.  Preprocess/Data Augmentation of the train and test data
### 4. Generate the train, validation and test images
### 5. Load the pretrained VGG16 model
### 6. Build the VGG16 model
### 7. Compile the VGG16 model
### 8. Train the VGG16 model
### 9. Plot graph for Accuracy and Loss over Epochs
### 10. Evaluate the ResNet50 model
### 11. Predict and display the metrics of the ResNet50 model
    
These steps are done in following files (one for each model):
- FishImgClassification_CNN.ipynb
- FishImgClassification_EfficientNetB0.ipynb
- FishImgClassification_InceptionV3.ipynb
- FishImgClassification_MobileNetV2.ipynb
- FishImgClassification_ResNet50.ipynb
- FishImgClassification_VGG16.ipynb

## Streamlit Fish Image Classifier Application:
Select a fish image from your drive in the file uploader box and click the "Predicted Fish Category" button. The App displays the predicted fish class\category to which this Fish image belongs.

#### To run the file:
Run the cells one by one in every ipynb model file
### To run the Streamlit Application:
Run the following command from the terminal:
streamlit run <Filepath>FishImgClassificationApp.py
