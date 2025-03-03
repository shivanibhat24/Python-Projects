# Digit Generation with GANs

This repository contains a script (`digitgen.py`) to train a Generative Adversarial Network (GAN) for generating handwritten digits based on the MNIST dataset. The model learns from the MNIST training dataset to produce images that resemble real handwritten digits.

## Features

- **Dataset**: Uses the MNIST dataset of handwritten digits.
- **Image Normalization**: Normalizes images to the [-1, 1] range for better GAN performance.
- **Model Architecture**: Implements a simple GAN using TensorFlow and Keras.
- **Training**: Trains the GAN to generate realistic digit images.

## Prerequisites

To run this script, you need the following Python packages:
- `tensorflow`
- `numpy`
- `matplotlib`
- `PIL`
- `imageio`

You can install them using pip:
```bash
pip install tensorflow numpy matplotlib pillow imageio

## Usage

a)Load and preprocess the dataset:

The script automatically loads the MNIST dataset with TensorFlow's tf.keras.datasets.mnist module.
It reshapes and normalizes the images to fit the GAN input requirements.

b)Define the GAN architecture:

The Generator model is designed to produce 28x28 images from random noise vectors (latent space of 100 dimensions).
The Discriminator model distinguishes real MNIST images from generated images.

c)Training the GAN:

The training function iteratively:
Generates images using the generator.
Evaluates them with the discriminator.
Updates both models to improve performance.
Training parameters such as BUFFER_SIZE (for shuffling) and BATCH_SIZE (for mini-batch size) are customizable within the script.

d)Run the script:

Simply run python digitgen.py in your terminal to start training the GAN.
As training progresses, the generator creates sample images of digits that can be displayed or saved.
