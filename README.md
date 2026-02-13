Classification and Captioning of Aircraft Damage Using Pretrained Models
Project Overview

This project focuses on detecting and classifying aircraft surface damage using deep learning and transfer learning techniques. The system performs binary classification (crack vs dent) and optionally generates descriptive captions for detected damage using a pretrained image captioning model.

The goal is to demonstrate practical application of pretrained convolutional neural networks and transformer-based vision-language models in a real-world inspection scenario.

Model Architecture
Damage Classification Model

Base model:

VGG16 (Keras)

weights = imagenet

include_top = False

input_shape = (224, 224, 3)

The VGG16 base is used as a frozen feature extractor.

Custom classification head:

Flatten

Dense(512, relu)

Dropout(0.3)

Dense(512, relu)

Dropout(0.3)

Dense(1, sigmoid)

All base layers are frozen (no fine-tuning). Only the custom dense layers are trained.

Training Configuration

Loss function: binary_crossentropy

Optimizer: Adam (learning_rate = 0.0001)

Epochs: 15

Batch size: 32

Input size: 224 × 224 × 3

Data preprocessing:

ImageDataGenerator(rescale = 1./255)

flow_from_directory with class_mode = binary

Train: shuffle = True

Validation/Test: shuffle = False

Reproducibility:

random.seed(42)

numpy seed = 42

tensorflow seed = 42

Evaluation

The model is evaluated using:

model.evaluate(test_generator)

Reported metrics:

Test Loss

Test Accuracy

To extend evaluation, precision, recall, and F1-score can be computed using sklearn.metrics by collecting predictions via model.predict().

Captioning Model (Optional Component)

This project also integrates image captioning using:

Salesforce/blip-image-captioning-base

Hugging Face Transformers

BlipProcessor

BlipForConditionalGeneration

A custom layer leverages tf.py_function to generate captions for aircraft damage images.

This demonstrates multimodal AI integration combining CNN-based classification with transformer-based caption generation.

Technologies Used

Python

TensorFlow / Keras

Transfer Learning

VGG16

Hugging Face Transformers

BLIP

NumPy