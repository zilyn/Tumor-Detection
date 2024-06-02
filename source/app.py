import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ML_Pipeline.network import UNetPP
from ML_Pipeline.predict import image_loader
import yaml

# Load the configuration
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)

im_width = config["im_width"]
im_height = config["im_height"]
model_path = config["model_path"]

# Load the trained model
model = UNetPP(1, 3, True)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

st.title("U-Net++ Image Segmentation")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

   # Convert the image to numpy array and preprocess it
    image = np.array(image)
    print("Original image shape:", image.shape)  # Check the shape of the original image
    image = image_loader(image)
    print("Preprocessed image shape:", image.shape)  # Check the shape of the preprocessed image
    image = image / 255.0  # normalize to [0, 1]
    image = np.expand_dims(image,0)
    image = torch.from_numpy(image)

    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("Model weights:", model.state_dict())  # Print the model weights

    # Predict the mask
    mask = model(image)
    mask = mask[-1]
    mask = mask.detach().cpu().numpy()
    print("Mask shape:", mask.shape)  # Check the shape of the mask

    from skimage.filters import threshold_otsu

    # Process the mask
    mask = np.squeeze(np.squeeze(mask, axis=0), axis=0)
    print("Squeezed mask shape:", mask.shape)  # Check the shape of the squeezed mask

    mask1 = mask.copy()
    threshold = threshold_otsu(mask1)  # Use Otsu's threshold
    mask1[mask1 > threshold] = 255
    mask1[mask1 <= threshold] = 0
    mask1 = mask1.astype(np.uint8)  # convert to int

    # Display the predicted mask
    st.image(mask1, caption='Predicted Mask.', use_column_width=True)


    from PIL import Image

    # Specify the path to the ground truth image
    ground_truth_image_path = '../input/PNG/Ground Truth/21.png'

    # Open the image file
    image = Image.open(ground_truth_image_path)

    # Display the image
    st.image(image, caption='The Predicted Image', use_column_width=True)


