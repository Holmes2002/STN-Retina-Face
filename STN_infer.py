from STN.model import  Advanced_STNet, Residual_STNet
import os
import torch
import numpy as np
from PIL import Image
import cv2
def convert_image_np(image):
    img = image.detach().cpu().numpy()
    denormalized_image = np.array(img*255, dtype = np.int32)
    return np.array(denormalized_image[0])

def resize_with_padding_landmark(image, target_size=(320, 320), fill_color=(0, 0, 0)):
    # Get the original size of the image
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate the new size while maintaining the aspect ratio
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))

    # Resize the image
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    # Create a new image with the target size and fill it with the fill color
    new_image = Image.new("RGB", target_size, fill_color)

    # Calculate paste position to center the resized image
    paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    paste_position = (0,0)
    new_image.paste(resized_image, paste_position)

    return new_image

if __name__ == '__main__':
    device = 'cuda'
    os.makedirs("STN_checkfolder", exist_ok = True)
    root = '/home1/data/vinhnguyen/Deepstream/video_hawkice_extract_car/LP'
    model = Advanced_STNet()
    model.load_state_dict(torch.load("/home/data2/congvu/checkpoint_base/38_258.pt"))
    model.to(device)

    for index, folder in enumerate(os.listdir(root)):
        for img_path in os.listdir(f"{root}/{folder}")[:5]:
            image = Image.open(f'{root}/{folder}/{img_path}')
            # ori_image = cv2.imread(f'{root}/{folder}/{img_path}')
            # ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
            input = resize_with_padding_landmark(image).convert("L")
            ori_image = np.array(input)
            input = torch.tensor(np.array(input)/255.).unsqueeze(0).unsqueeze(0).to(device).float()
            output, _ = model(input)
            for i in range(output.shape[0]):
                output_ = convert_image_np(output[i])
                # ori_image = cv2.resize(ori_image, (320, 320))
                # print(ori_image.shape, output_.shape)
                vis = np.concatenate((ori_image, output_), axis=1)
                cv2.imwrite(f'STN_checkfolder/{index}_{img_path}', vis)

