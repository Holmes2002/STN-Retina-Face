import cv2 
import torch
import os
from PIL import Image, ImageFile
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from STN.transform import random_augment_test
from copy import deepcopy
from STN.utils import Affine_aug, Perspective_aug
import random


from torch.nn.functional import affine_grid, grid_sample
def convert_image_np(image):

    img=image.detach().cpu()
    denormalize = transforms.Normalize(mean=[0.], std=[1.])
    denormalized_tensor = denormalize(image)
    denormalized_image = transforms.ToPILImage()(denormalized_tensor)
    return np.array(denormalized_image)

# Function to apply affine transformation to an image using a given affine matrix
def apply_affine_to_image(image, affine_matrix):
    # Ensure the affine matrix is in the correct format
    if affine_matrix.shape != (2, 3):
        raise ValueError("Affine matrix must be of shape (2, 3)")

    # Convert the image to a tensor
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)

    # Prepare affine grid
    grid = affine_grid(torch.tensor(affine_matrix).unsqueeze(0), image_tensor.size(), align_corners=False)
    transformed_image = grid_sample(image_tensor, grid, align_corners=False)

    # Convert tensor back to image
    transformed_image = transformed_image.squeeze().numpy()
    return transformed_image


def resize_with_padding_label(image, target_size=(320, 320), fill_color=(0, 0, 0)):
    # Get the original size of the image
    # gray_value = random.randint(0, 128)
    # gray_value = 128
    # fill_color = (gray_value, gray_value, gray_value)

    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate the new size maintaining the aspect ratio
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))

    # Resize the image
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    # Create a new image with the target size and fill it with the fill color
    new_image = Image.new("RGB", target_size, fill_color)

    # Paste the resized image onto the center of the new image
    paste_position = ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2)
    # paste_position = (0,0)
    new_image.paste(resized_image, paste_position)

    # Calculate the new corner landmarks based on the resized image
    top_left = (paste_position[0], paste_position[1])
    top_right = (paste_position[0] + new_size[0], paste_position[1])
    bottom_left = (paste_position[0], paste_position[1] + new_size[1])
    bottom_right = (paste_position[0] + new_size[0], paste_position[1] + new_size[1])

    # Store the four corner landmarks
    corner_landmarks = [top_left, top_right, bottom_left, bottom_right]
    # draw = ImageDraw.Draw(new_image)
    # for x, y in corner_landmarks:
    #     radius = 5  # You can adjust the radius as needed
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))  # Red corner points

    # # Save the image if save_path is provided
    # new_image = new_image.convert("L")
    # new_image.save('sample.jpg')
    # assert False

    return new_image, np.array(corner_landmarks)
def resize_with_padding_landmark(image, landmarks, target_size=(320, 320), fill_color=(0, 0, 0)):
    landmarks = np.array(landmarks).reshape((4,2))
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
    # paste_position = (0,0)
    new_image.paste(resized_image, paste_position)

    # Adjust landmarks to the new size and position
    scaled_landmarks = []
    for x, y in landmarks:
        # Scale the points based on the ratio
        new_x = int(x * ratio) + paste_position[0]
        new_y = int(y * ratio) + paste_position[1]

        # Clamp the points within the target size bounds
        new_x = max(0, min(new_x, target_width - 1))
        new_y = max(0, min(new_y, target_height - 1))

        scaled_landmarks.append((new_x, new_y))
    # draw = ImageDraw.Draw(new_image)
    # for x, y in scaled_landmarks:
    #     radius = 5  # You can adjust the radius as needed
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))  # Red corner points

    # # Save the image if save_path is provided
    # new_image.save('sample_ori.jpg')
    # assert False

    return new_image, scaled_landmarks

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size  ,train_transform_aug, training = True, visualize_train = False):
        # self.train_transform_target = train_transform_target
        self.path_labels, self.path_images, self.landmarks = [], [], []
        if training and not visualize_train:
            for folder in os.listdir(root):
                if '2' in folder : continue
                for img_path in os.listdir(f'{root}/{folder}/labels')[:-30]:
                    self.path_labels.append(f'{root}/{folder}/labels/{img_path}')
                    self.path_images.append(f'{root}/{folder}/images/{img_path}')
                    txt_file = img_path.replace('jpg', 'txt')
                    landmark_ = [float(i) for i in open(f'{root}/{folder}/texts/{txt_file}').read().splitlines()[0].split()]
                    self.landmarks.append(landmark_)
        elif visualize_train:
            for folder in os.listdir(root):
                if '2' in folder : continue
                tmp_list = os.listdir(f'{root}/{folder}/labels')
                random.shuffle(tmp_list)
                for img_path in tmp_list[:20]:
                    self.path_labels.append(f'{root}/{folder}/labels/{img_path}')
                    self.path_images.append(f'{root}/{folder}/images/{img_path}')
                    txt_file = img_path.replace('jpg', 'txt')
                    landmark_ = [float(i) for i in open(f'{root}/{folder}/texts/{txt_file}').read().splitlines()[0].split()]
                    self.landmarks.append(landmark_)

        else:
            for folder in os.listdir(root):
                if folder != '4':
                    if '2' in folder : continue
                    for img_path in os.listdir(f'{root}/{folder}/labels')[-30:]:
                        self.path_labels.append(f'{root}/{folder}/labels/{img_path}')
                        self.path_images.append(f'{root}/{folder}/images/{img_path}')
                        txt_file = img_path.replace('jpg', 'txt')
                        landmark_ = [float(i) for i in open(f'{root}/{folder}/texts/{txt_file}').read().splitlines()[0].split()]
                        self.landmarks.append(landmark_)
                else:
                    for img_path in os.listdir(f'{root}/{folder}/labels'):
                        self.path_labels.append(f'{root}/{folder}/labels/{img_path}')
                        self.path_images.append(f'{root}/{folder}/images/{img_path}')
                        txt_file = img_path.replace('jpg', 'txt')
                        landmark_ = [float(i) for i in open(f'{root}/{folder}/texts/{txt_file}').read().splitlines()[0].split()]
                        self.landmarks.append(landmark_)

        self.training = training
        self.image_size = image_size
        self.train_transform_aug = train_transform_aug
    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        image = Image.open(self.path_images[idx]).convert("RGB")
        label = Image.open(self.path_labels[idx]).convert("RGB")
        landmark = self.landmarks[idx]

        image, landmark  = resize_with_padding_landmark(image, landmark)
        label, landmark_target = resize_with_padding_label(label)

        if self.training:
            image, label = random_augment_test(image, label)   
            image = image.convert('L')  # Convert image to grayscale
            label = label.convert('L')  # Convert label to grayscale, if needed
            image_aug = torch.tensor([np.array(image, dtype = np.float32)])
            
            # pseudo_label = np.array(image_aug[0]*255, dtype = np.int32)
            image_aug = self.train_transform_aug(image)
            image_aug = image_aug.unsqueeze(dim=0)
            if random.random() <= 0.5:
                image_aug, theta = Perspective_aug(image_aug, landmark = landmark, target_landmarks = landmark_target, tmp_img = cv2.imread(self.path_images[idx]))
            else:
                image_aug, theta = Affine_aug(image_aug, landmark = landmark, target_landmarks = landmark_target, tmp_img = cv2.imread(self.path_images[idx]))
                image_aug, theta = image_aug[0], theta
            # pseudo_label = np.array((image_aug[0])*255, dtype = np.int32)
            # assert False
            # image_aug, theta = image_aug, 0
        else: 
            image = image.convert('L')  # Convert image to grayscale
            label = label.convert('L')  # Convert label to grayscale, if needed
            image_aug = self.train_transform_aug((image))
            H, W = 320, 320
            image_aug = image_aug.unsqueeze(dim=0)
            if random.random() <= 0.5:
                image_aug, theta = Perspective_aug(image_aug, landmark = landmark, target_landmarks = landmark_target, tmp_img = cv2.imread(self.path_images[idx]))
                
            else:
                image_aug, theta = Affine_aug(image_aug, landmark = landmark, target_landmarks = landmark_target, tmp_img = cv2.imread(self.path_images[idx]))

                image_aug, theta = image_aug[0], theta

            # # Normalization điểm nguồn và điểm đích về không gian [-1, 1]
            # src_point_normalized = (landmark_target / np.array([W, H]) * 2) - 1
            # des_point_normalized = (np.array(landmark) / np.array([W, H]) * 2) - 1

            # # Thêm 1 để tạo tọa độ đồng nhất (homogeneous coordinates)
            # src_point_augmented = np.hstack([src_point_normalized, np.ones((src_point_normalized.shape[0], 1))])

            # # Tính toán ma trận affine bằng phương pháp bình phương tối thiểu
            # A, _, _, _ = np.linalg.lstsq(src_point_augmented, des_point_normalized, rcond=None)

            # # Ma trận affine cần được chuyển đổi từ 3x2 thành 2x3 cho PyTorch
            # affine_matrix = A.T
            # affine_matrix = np.array(affine_matrix, dtype = np.float32)
            # new_affine_matrix_tensor = torch.tensor(affine_matrix, dtype=torch.float32)

        label = self.train_transform_aug(label)
        if not self.training:
            return image_aug, label, theta
        return image_aug, label, theta
    
def get_dataloader_train( root, image_size, train_transform_aug,batch_size, shuffle=False, training = True, visualize_train = False):
    ds_avatarsearch = SampleDataset( root, image_size, train_transform_aug, training, visualize_train)

    # Use dataloader with num_workers is 0 (not use num_workers)
    dataloader = torch.utils.data.DataLoader(ds_avatarsearch, batch_size=batch_size, 
                                             shuffle=shuffle)
    
    return dataloader



