# Sketches_to_Face_CycleGANS

## Research Paper:
[Report](Report.pdf)



## Project Overview 
This project implements a CycleGAN model for image-to-image translation between real face images and their corresponding sketches using the Person Face Sketches dataset. The model is capable of converting face photos to sketches and vice versa, following the original CycleGAN paper's approach.

## Objective
The primary goal of this project is to learn a mapping between two domains:

Domain A: Sketches of faces
Domain B: Realistic photos of faces

At test time, the trained model can:
Generate a realistic face from a given sketch
Generate a sketch from a given real face image

## Libraries 
PyTorch, matplotlib, numpy, os, random

## Model Architecture
### Generators:
1) Generator_A2B: Sketch → Real Face
2) Generator_B2A: Real Face → Sketch
   
### Discriminators:
1) Discriminator_A: Distinguishes between real vs generated sketches
2) Discriminator_B: Distinguishes between real vs generated face images

The architecture includes convolutional and transpose convolutional layers with ReLU and LeakyReLU activations, and is trained using adversarial and cycle-consistency losses.


## Setup and Installination 
1) Clone the repository and navigate to the project folder.
2) Install all required libraries listed in the requirements.txt
3) Download and prepare the Person Face Sketches dataset.
4) Organize the dataset into train/photos, train/sketches, val/photos, val/sketches, and test/ folders accordingly.
5) Run the training script to start training the CycleGAN model.
6) Launch the Flask application provided in app.py.
7) Open a web browser and go to http://localhost:5000.



## Datasets
[Person Face Sketches Dataset](https://www.kaggle.com/datasets/almightyj/person-face-sketches)



## Results 
The CycleGAN model effectively performed image-to-image translation between sketches and real face images. During training, the model learned to generate visually convincing outputs in both directions—sketch to photo and photo to sketch—while preserving identity features. Over multiple epochs, qualitative improvements were observed with smoother textures and more realistic facial details. The cyclic consistency loss helped maintain structural coherence, ensuring that converting back and forth between domains resulted in minimal loss of information. Overall, the model produced high-quality, perceptually accurate translations.


![Cycle_Image](https://github.com/user-attachments/assets/3c7c8407-dc9e-4de8-b28a-c3b67a698ac4)

![INTERFACE}(https://github.com/user-attachments/assets/875be97b-7e69-4aa0-a094-0ee80fef6c97)



