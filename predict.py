import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from source.network import UNetPP
from argparse import ArgumentParser
from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from albumentations.core.composition import Compose
from PIL import Image

val_transform = Compose([
    geometric.resize.Resize(256, 256),
    transforms.Normalize(),
])

def image_loader(image_name):
    img = cv2.imread(image_name)
    img = val_transform(image=img)["image"]
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)

    return img

def predict(image_path):
    with open("config.yaml") as f:
        config = yaml.load(f)

    im_width = config["im_width"]
    im_height = config["im_height"]
    model_path = config["model_path"]

    model = UNetPP(1, 3, True)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    image = image_loader(image_path)
    image = np.expand_dims(image,0)
    image = torch.from_numpy(image)

    if torch.cuda.is_available():
        image = image.to(device="cuda")
    mask = model(image)
    mask = mask[-1]
    mask = mask.detach().cpu().numpy()
    mask = np.squeeze(np.squeeze(mask, axis=0), axis=0)
    mask1 = mask.copy()
    mask1[mask1 > -2.5] = 255
    mask1[mask1 <= -2.5] = 0
    mask1 = cv2.resize(mask1, (im_width, im_height))

    return mask1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_img", default="input/PNG/Original/15.png", help="path to test image")

    opt = parser.parse_args()
    mask = predict(opt.test_img)

    plt.imsave("prediction.png", mask, cmap="gray")