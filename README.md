# Semantic Segmentation to Color Image Conversion using Pix2Pix

This repository contains code for a Pix2Pix-based project that converts semantic segmented images into colored RGB images. The project involves a Generator and a Discriminator trained using PyTorch.

## Overview

This project leverages Pix2Pix, a conditional GAN architecture, to convert semantic segmented images to colored RGB images. The repository includes:

- `generator_model.py`: A PyTorch implementation of the Generator, responsible for converting segmented images to colored ones.
- `discriminator_model.py`: Implementation of the Discriminator used to evaluate the authenticity of the generated images.
- `train.py`: Code for training the Generator and Discriminator models using a defined dataset.
- `dataset.py`: Custom dataset processing and augmentation for model training.
- `config.py` and `utils.py`: Configuration files, model checkpointing, and other utilities used during training and evaluation.

## Requirements

- Python 3
- PyTorch
- torchvision
- NumPy
- tqdm
- OpenCV
- Albumentations

## Architecture
- Generator: The Generator model consists of a series of down-sampling and up-sampling blocks, converting segmented images to colored representations.
- Discriminator: The Discriminator evaluates the authenticity of generated images compared to real ones.
  
  ## Results
| Input Image | Output Image |
|---------|---------|
| ![Input Image](https://i.imgur.com/9gkAMYy.png) | ![Output Image](https://i.imgur.com/d8av3qM.png) |

This is the result for 150 epochs of training for the model. The model produces better results as it is trained for longer

## Acknowledement
This project utilizes the Pix2Pix architecture developed by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros from UC Berkeley. We acknowledge and appreciate their groundbreaking work on image-to-image translation using conditional adversarial networks. For more details about Pix2Pix, please refer to their paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A.
