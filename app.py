import sys
import os
import glob
import re
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, redirect, url_for, request, render_template, make_response
from werkzeug.utils import secure_filename
import cv2

# Define a Flask app
app = Flask(__name__ , template_folder='template')
latent_size=128
generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)

# Load the generator model from the checkpoint
checkpoint = torch.load("C:/Users/LENOVO/OneDrive/Documents/Bureau/Deploy DL model/G.pth", map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint)
generator = generator.to(torch.device('cpu'))
generator.eval()

@app.route('/generate_image', methods=['GET'])
def generate_image():
    # Generate a random latent tensor
    latent_tensor = torch.randn(1, 128, 1, 1)

    # Generate an image using the generator
    with torch.no_grad():
        generated_image = generator(latent_tensor)

    # Convert the image tensor to a numpy array
    generated_image = generated_image.squeeze().permute(1, 2, 0).numpy()
    
        # Scale the pixel values from [-1, 1] to [0, 255]
    generated_image = ((generated_image + 1) / 2) * 255

    # Convert to uint8 data type
    generated_image = generated_image.astype(np.uint8)

    
    return generated_image


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Generate and return an anime face image
        generated_image = generate_image()

        # Convert the image to base64 string
        _, encoded_image = cv2.imencode('.png', generated_image)
        base64_image = encoded_image.tobytes()

        # Create a response with the image data
        response = make_response(base64_image)
        response.headers.set('Content-Type', 'image/png')
        return response
    else:
        # Handle GET request, redirect to the main page
        return redirect(url_for('index'))

    


if __name__ == '__main__':
    app.run(debug=True)
