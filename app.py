import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO
import torch.nn as nn
import torch.optim as optim

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the layers of the generator
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # Normalize to [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

IMAGE_SIZE = 256

# Function to remove 'module.' prefix from keys in state_dict
def remove_module_prefix(state_dict):
    # Create new OrderedDict without 'module.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_A2B = Generator().to(device)  # Sketch to Photo
generator_B2A = Generator().to(device)  # Photo to Sketch

# Load the state dicts and handle 'module.' prefix if needed
state_dict_A2B = torch.load('generator_A2B.pth', map_location=device)
state_dict_B2A = torch.load('generator_B2A.pth', map_location=device)

# Remove 'module.' prefix if necessary and load the models
generator_A2B.load_state_dict(remove_module_prefix(state_dict_A2B))
generator_B2A.load_state_dict(remove_module_prefix(state_dict_B2A))

generator_A2B.eval()
generator_B2A.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Reverse normalization to convert tensor to image
def reverse_transform(tensor):
    tensor = (tensor + 1) / 2
    img = transforms.ToPILImage()(tensor.squeeze(0))
    return img

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload and process image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        img = Image.open(file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Choose model based on user selection
        if request.form['conversion'] == 'sketch_to_photo':
            output = generator_A2B(img_tensor)
        else:
            output = generator_B2A(img_tensor)

        output_img = reverse_transform(output.cpu().detach())
       
        # Save the output image to a BytesIO object to send it as a response
        img_io = BytesIO()
        output_img.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
