# Convolutional Neural Network from Scratch
## Project Overview
This project is focused on building a Convolutional Neural Network (CNN) from scratch using Python. The goal is to understand mechanics of CNNs and this project is intended for educational purposes, to gain a deeper understanding of how CNNs work and how they can be applied image classification tasks.
## Features
- **Custom CNN Architecture**: Implementation of a CNN architecture from the ground up.
- **Save/Load Model**: ability to save and load the trained model. 
- **Image Classification**: Ability to classify images in real time (hand gestures regconition).
- **Data Collection**: Scripts for collecting data (available for custom classification).
## Implementation Details
- Convolutional layer
- Activation Function
- Pooling Layer
- Fully Connected (Dense) Layer
## Optimization Implementation
- **He initialization**: this method is for weights initialization that keeps the scale of gradients roughly the same in all layers. It is particularly effective for layers with ReLU activation.
- **Xavier/Glorot initialization**: in layers where the 'sigmoid' activation function is used, this method calculates the bounds of the uniform distribution for initializing the weights based on the number of input and output nodes of each layer.
- **Adam Optimization**: Adam (Adaptive Moment Estimation) is used for updating network weights iteratively based on training data. This is the better method than the default one (Gradient Descent)
