# ALZHEIMER DISEASE CLASSIFICATION

This project builds a Convolutional Neural Network (CNN) to classify brain MRI images into four stages of Alzheimer’s disease progression. It uses TensorFlow, Keras, and SMOTE for deep learning and class imbalance handling.

# Dataset

The dataset used is the Alzheimer’s Dataset (4 Class of Images) from Kaggle:
🔗 Alzheimer’s Dataset on Kaggle

# Classes

🟢 NonDemented

🟡 VeryMildDemented

🟠 MildDemented

🔴 ModerateDemented

# Workflow Overview

## Data Preprocessing

Loaded and merged training & testing data into a working directory.

Applied ImageDataGenerator for data augmentation (brightness, zoom, flip).

Resized all images to 176×176 pixels.

Balanced the dataset using SMOTE.

## Model Architecture



Convolutional, MaxPooling, and BatchNormalization layers

Dense + Dropout layers for regularization

softmax activation for multi-class classification

Early stopping and custom callbacks used for training efficiency.

## Training & Evaluation

Trained for up to 100 epochs with validation monitoring.

## Evaluated using metrics:

Accuracy (CategoricalAccuracy)

AUC (Area Under Curve)

F1 Score (TensorFlow Addons)

Visualized accuracy, loss, and AUC trends.

## Performance Metrics

Confusion Matrix

Classification Report

Balanced Accuracy Score (BAS)

Matthews Correlation Coefficient (MCC)

## Model Saving

Saved trained model as alzheimer_cnn_model.h5

Model architecture plotted using plot_model()

## Dependencies

tensorflow

tensorflow-addons

keras

imblearn

seaborn

numpy

pandas

matplotlib

Pillow

scikit-learn

## How to Run
### Clone this repository

git clone https://github.com/<your-username>/alzheimers-disease-classification.git

cd alzheimers-disease-classification

### Run the notebook
jupyter notebook Alzheimer's_Classification.ipynb

# Future Work
Integration of InceptionV3 / EfficientNet for improved feature extraction.

Deploying as a web-based diagnostic tool.

Extending to 3D MRI scans for more robust predictions.
