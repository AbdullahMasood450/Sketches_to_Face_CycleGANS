# Sketches_to_Face_CycleGANS

## Research Paper:
[Report](Report.pdf)



## Project Overview 
This project implements a CycleGAN-style model to perform image-to-image translation between human face photographs and their corresponding sketches. The model is capable of converting:
A real face image into a sketch, and
A sketch into a realistic face image.
The training was done end-to-end on the Person Face Sketches dataset. At test time, the model supports real-time generation of converted images through an intuitive user interface.

## Tools and Technologies Used
Programming Language: Python 
Google Colab for training

PyTorch for deep learning

Matplotlib & PIL for visualization and image handling

Flask for deploying the real-time UI

## Libraries 
torch, torchvision

PIL

matplotlib

numpy

os, random

## Techniques Used 
Conditional GAN (cGAN) for image-to-image translation

Cycle Consistency Loss (implicitly through L1 reconstruction)

Data normalization and preprocessing

Model checkpointing every epoch

Grayscale and RGB image handling

Adversarial and reconstruction losses


## Setup and Installination 
1) Clone the repository
2) Run the .ipynb file
3) Install the required libraries
4) Install the Dataset from the below link
5) Train the GAN and generate images

##Datasets
[Person Face Sketches Dataset](https://www.kaggle.com/datasets/almightyj/person-face-sketches)



## Results 
The CycleGAN model effectively performed image-to-image translation between sketches and real face images. During training, the model learned to generate visually convincing outputs in both directions—sketch to photo and photo to sketch—while preserving identity features. Over multiple epochs, qualitative improvements were observed with smoother textures and more realistic facial details. The cyclic consistency loss helped maintain structural coherence, ensuring that converting back and forth between domains resulted in minimal loss of information. Overall, the model produced high-quality, perceptually accurate translations.


![Cycle_Image](https://github.com/user-attachments/assets/3c7c8407-dc9e-4de8-b28a-c3b67a698ac4)




