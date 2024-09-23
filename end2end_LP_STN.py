import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import os
from STN.dataset import *
from STN.model import STNet
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for preprocessing
train_transform_aug = transforms.Compose([
    # Apply transformations like random rotation, resize, etc., if needed
    # transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_path):
    """Preprocess a single image from the given path."""
    # Load the image using OpenCV
    image = Image.open(image_path)  
    image = image.resize((112,112)) 
    # Apply the defined transformations
    transformed_image = train_transform_aug(image)
    # Add a batch dimension
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image
def convert_image_np(image):

    img=image.detach().cpu()
    denormalize = transforms.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5])
    denormalized_tensor = denormalize(image)
    denormalized_image = transforms.ToPILImage()(denormalized_tensor)
    return np.array(denormalized_image)
def evaluate(list_img):
    # Initialize the model
    model = STNet()
    # Load the model weights
    model.load_state_dict(torch.load('/home/data2/congvu/checkpoint/99_17.pt'))
    model.to(device)
    model.eval()

    # Preprocess and prepare data
    preprocessed_images = []
    for img_path in list_img:
        preprocessed_image = preprocess_image(img_path)
        preprocessed_images.append(preprocessed_image)
    
    # Convert list of preprocessed images to a batch tensor
    data = torch.cat(preprocessed_images).to(device)

    # Perform inference
    output = model(data)

    # Process and save results
    idx = 0
    for i in range(output.shape[0]):
        idx += 1
        output_ = convert_image_np(output[i])
        data_ = convert_image_np(data[i])
        # Convert RGB to BGR for visualization
        output_ = output_[:, :, ::-1]
        data_ = data_[:, :, ::-1]
        vis = np.concatenate((data_, output_), axis=1)  # Concatenate input and output side by side
        os.makedirs(f"results", exist_ok=True)
        cv2.imwrite(f"results/{idx}.png", vis)
if __name__ == '__main__':
    list_img = [f"/home1/data/congvu/TMP/cropped_LPs/{path}" for idx, path in enumerate(os.listdir("/home1/data/congvu/TMP/cropped_LPs")) if idx % 10 == 0][:128]
    evaluate(list_img)
