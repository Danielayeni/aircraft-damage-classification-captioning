**Aircraft Damage Classification and Captioning**
**Overview**

This project implements a deep learning pipeline for automated aircraft damage inspection. It performs binary image classification (crack vs dent) using transfer learning with a pretrained VGG16 model.

An optional image captioning module integrates a transformer-based BLIP model to generate descriptive summaries of detected damage, demonstrating multimodal AI integration in a real-world computer vision application.

**Problem Statement**

Manual aircraft damage inspection can be time-consuming and subjective. This project explores how pretrained deep learning models can assist in:

Classifying types of structural surface damage

Automating visual inspection workflows

Generating descriptive summaries of damage regions

**Model Architecture
1. Classification Model

Base Model**

VGG16 (Keras)

weights = "imagenet"

include_top = False

input_shape = (224, 224, 3)

The pretrained VGG16 network is used as a frozen feature extractor.

Custom Classification Head

Flatten

Dense(512, activation = relu)

Dropout(0.3)

Dense(512, activation = relu)

Dropout(0.3)

Dense(1, activation = sigmoid)

All VGG16 base layers are frozen. Only the custom dense layers are trained.

**2. Captioning Model (Optional Component)**

Model: Salesforce/blip-image-captioning-base

Framework: Hugging Face Transformers

Components: BlipProcessor, BlipForConditionalGeneration

The captioning module uses a transformer-based vision-language model to generate descriptive summaries of aircraft damage images.

**Training Configuration**

Loss Function: binary_crossentropy

Optimizer: Adam (learning_rate = 0.0001)

Epochs: 15

Batch Size: 32

Input Size: 224 × 224 × 3

**Data Preprocessing**

ImageDataGenerator(rescale = 1./255)

flow_from_directory with class_mode = "binary"

Train shuffle = True

Validation/Test shuffle = False

**Reproducibility**
random.seed(42)

numpy random seed = 42

tensorflow random seed = 42


**Evaluation**

The model is evaluated using:

model.evaluate(test_generator)

Reported metrics:

Test Loss

Test Accuracy

Precision, Recall, and F1-score can be computed using sklearn.metrics by generating predictions via model.predict().

**Technologies Used**

Python

TensorFlow / Keras

Transfer Learning

VGG16

Hugging Face Transformers

BLIP

NumPy

Scikit-learn

**How to Run**

Install dependencies:

pip install -r requirements.txt

Prepare dataset in the specified directory structure.

Run the classification script:

python Classification_and_Captioning_Aircraft_Damage_Using_Pretrained_Models.py
Key Concepts Demonstrated

Transfer learning with pretrained CNN models

Feature extraction using frozen convolutional bases

Binary image classification

Multimodal AI integration (Vision + Language)

Reproducible training setup

