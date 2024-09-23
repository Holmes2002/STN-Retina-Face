from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from PIL import Image, ImageDraw
import torch.nn.functional as F
import cv2
from affine import Affine

def resize_with_padding(image, target_size=(320, 320), fill_color=(0, 0, 0)):
    # Get the original size of the image
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
    new_image.paste(resized_image, paste_position)

    return new_image


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):

        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:

                
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

import torch
import random
import math
import torch.nn.functional as F


def affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    grid = F.affine_grid(theta, size, align_corners=align_corners)
    return grid
def Affine_aug(im, landmark, target_landmarks, tmp_img, size=(320, 320)):


    landmark = np.array(landmark).reshape((4, 2))
    target_landmarks = np.array(target_landmarks).reshape((4, 2))

    
    # Ensure `im` is a torch tensor and has the right shape (N, C, H, W)
    if len(im.shape) == 3:
        im = im.unsqueeze(0)  # Add batch dimension if missing

    # Get image size for normalization
    _, _, H, W = im.shape

    # Random augmentation parameters
    angle = random.uniform(-50,50)  # Random angle between -20 and 20 degrees
    angle_rad = math.pi * angle / 180  # Convert to radians

    translation_x = random.uniform(-0.01, 0.01)  # Normalize to [-1, 1]
    translation_y = random.uniform(-0.01, 0.01)  # Normalize to [-1, 1]

    # Scaling
    scaling_x = random.uniform(0.8, 1.2)
    scaling_y = random.uniform(0.8, 1.2)

    # Shearing
    shearing_x = random.uniform(-0.1, 0.1)
    shearing_y = random.uniform(-0.1, 0.1)

    # translation_x = 0  # Normalize to [-1, 1]
    # translation_y = 0  # Normalize to [-1, 1]
    # angle = -15
    # ori_img = np.array(im[0][0]*255, dtype = np.int32)
    # cv2.imwrite('ori_img.jpg', ori_img)
    # # Scaling
    # scaling_x = 1
    # scaling_y = 1

    # # Shearing
    # shearing_x = 0
    # shearing_y = 0

    aug_affine_matrix = (
    Affine.translation(translation_x, translation_y)  # Translation
    * Affine.scale(scaling_x, scaling_y)             # Scaling
    * Affine.shear(math.degrees(shearing_x), math.degrees(shearing_y))  # Shearing (converted to degrees)
    * Affine.rotation(angle)  # Rotation
)      
    aug_affine_matrix = np.array(aug_affine_matrix[:6], dtype = np.float32).reshape((2,3))
    # print(affine_transformation)
    # assert False

    # Convert affine matrix to PyTorch tensor and reshape for PyTorch
    aug_affine_matrix_tensor = torch.tensor(aug_affine_matrix, dtype=torch.float32)
    aug_affine_matrix_tensor = aug_affine_matrix_tensor.unsqueeze(0)  # Make it 2x3 and batch it

    # Create affine grid and apply grid_sample
    rotated_grid = F.affine_grid(aug_affine_matrix_tensor, im.shape, align_corners=False)
    rotated_image = F.grid_sample(im, rotated_grid, align_corners=False)


    # print(landmark.shape, aug_affine_matrix.shape)
    # Step 1: Convert affine matrix to 3x3

    # des_point_normalized = landmark
    des_point_normalized = (landmark / np.array([W, H]) * 2) - 1

    # Convert aug_affine_matrix to 3x3
    affine_matrix_3x3 = np.vstack([aug_affine_matrix, [0, 0, 1]])

    # Invert the affine matrix
    inv_affine_matrix = np.linalg.inv(affine_matrix_3x3)

    # Add 1 to create homogeneous coordinates
    des_point_homogeneous = np.hstack([des_point_normalized, np.ones((des_point_normalized.shape[0], 1))])

    # Apply the inverse affine matrix to the normalized destination points
    src_point_homogeneous = des_point_homogeneous @ inv_affine_matrix.T

    # Extract the source points and de-normalize back to original scale
    src_point_normalized = src_point_homogeneous[:, :2]
    # source_points = src_point_normalized
    source_points = ((src_point_normalized + 1) * np.array([W, H]) / 2)

    source_points[:, 0] = np.clip(source_points[:, 0], 0, W)
    source_points[:, 1] = np.clip(source_points[:, 1], 0, H)

    # rotated_image_np = np.array(rotated_image[0][0].numpy() * 255, dtype = np.int32)
    # ori = np.array(im[0].numpy() * 255, dtype = np.int32)
    # cv2.imwrite('sample.jpg', rotated_image_np)
    # cv2.imwrite('ori.jpg', ori)
    # assert False
    # Convert landmarks to homogeneous coordinates

    # landmark_center = landmark - np.array([W / 2, H / 2])
    # landmarks_homo = np.concatenate([landmark_center, np.ones((landmark_center.shape[0], 1))], axis=1)
    # aug_affine_matrix = np.vstack([aug_affine_matrix, [0, 0, 1]])  # Shape (3, 3)
    # # Apply affine transformation to landmarks
    # new_landmarks = np.dot(landmarks_homo, aug_affine_matrix)
    # # print(new_landmarks)

    # new_landmarks = new_landmarks[:,:2]
    # new_landmarks += np.array([W / 2, H / 2])
    # # print(new_landmarks)
    # # Rescale landmarks from normalized to pixel coordinates
    # # Clamp landmarks to image bounds
    # new_landmarks[:, 0] = np.clip(new_landmarks[:, 0], 0, W)
    # new_landmarks[:, 1] = np.clip(new_landmarks[:, 1], 0, H)

    # # Convert image back to PIL format
    # rotated_image_np = rotated_image[0][0].numpy() * 255
    # center = np.array([W / 2, H / 2])
    # check_landmark = rotate_landmarks(landmark, -angle, center)
    # check_landmark[:, 0] = np.clip(check_landmark[:, 0], 0, W)
    # check_landmark[:, 1] = np.clip(check_landmark[:, 1], 0, H)


    # Draw landmarks
    # rotated_image_np = rotated_image[0][0].numpy() * 255
    # new_image = Image.fromarray(rotated_image_np.astype(np.uint8)).convert("RGB")
    # draw = ImageDraw.Draw(new_image)
    # for x, y in landmark:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 255))  # Red points
    # for x, y in source_points:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))  # Red points
    # for x, y in target_landmarks:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))  # Green points

    # new_image.save('sample_2.jpg')
    # assert False
    # Calculate affine matrix to transform augmented image back


    # draw = ImageDraw.Draw(new_image)
    # for x, y in check_landmark:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 255))  # Red points
    # for x, y in target_landmarks:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))  # Red points
    # for x, y in source_points:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))  # Red points
    # new_image.save('sample_2.jpg')
    # print(check_landmark)
    # print(target_landmarks)
    # assert False

    # new_affine_matrix, _ = cv2.estimateAffinePartial2D(check_landmark, target_landmarks, method=cv2.LMEDS)
    # print(new_affine_matrix)
    H, W = 320, 320

    # Normalization điểm nguồn và điểm đích về không gian [-1, 1]
    src_point_normalized = (target_landmarks / np.array([W, H]) * 2) - 1
    des_point_normalized = (source_points / np.array([W, H]) * 2) - 1

    # Thêm 1 để tạo tọa độ đồng nhất (homogeneous coordinates)
    src_point_augmented = np.hstack([src_point_normalized, np.ones((src_point_normalized.shape[0], 1))])

    print(src_point_augmented.shape)
    print(des_point_normalized.shape)
    assert False
    # Tính toán ma trận affine bằng phương pháp bình phương tối thiểu
    A, _, _, _ = np.linalg.lstsq(src_point_augmented, des_point_normalized, rcond=None)

    # Ma trận affine cần được chuyển đổi từ 3x2 thành 2x3 cho PyTorch
    affine_matrix = A.T
    affine_matrix = np.array(affine_matrix, dtype = np.float32)

    # In ra ma trận affine để kiểm tra
    # print("Affine Matrix: ", affine_matrix)
    # new_affine_matrix = torch.tensor(affine_matrix, dtype=torch.float32).unsqueeze(0)  # Thêm batch dimension
    # rotated_grid = F.affine_grid(new_affine_matrix, rotated_image.size(), align_corners=False)
    # rotated_image = F.grid_sample(rotated_image, rotated_grid, align_corners=False)
    # rotated_image_np = rotated_image[0][0].numpy() * 255.0  # Chuyển đổi về định dạng HxWxC và scale về [0, 255]
    # rotated_image_np = rotated_image_np.astype(np.uint8)
    # new_image = Image.fromarray(rotated_image_np).convert("RGB")

    # # Lưu hình ảnh đã được biến đổi
    # new_image.save('sample.jpg')
    # assert False

    # assert False
    # perspective_matrix = cv2.getPerspectiveTransform(new_landmarks, target_landmarks)
    # perspective_matrix /= perspective_matrix[2, 2]

    # # Extract the affine matrix by taking only the first two rows and the first three columns
    # affine_matrix = perspective_matrix[:2, :3]
    # print(perspective_matrix)
    # print(affine_matrix)
    # new_affine_matrix_tensor = torch.tensor(new_affine_matrix, dtype=torch.float32)


    return rotated_image, torch.tensor(affine_matrix, requires_grad=True)

def Perspective_aug(im, landmark, target_landmarks, tmp_img, size=(320, 320)):

    landmark = np.array(landmark).reshape((4, 2))
    target_landmarks = np.array(target_landmarks).reshape((4, 2))

    # Ensure `im` is a torch tensor and has the right shape (N, C, H, W)
    if len(im.shape) == 3:
        im = im.unsqueeze(0)  # Add batch dimension if missing

    # Get image size for normalization
    _, _, H, W = im.shape

    # Convert the torch tensor to a NumPy array for OpenCV processing
    im_np = im[0].permute(1, 2, 0).numpy()  # (H, W, C)

    # Random perturbation on target landmarks (for augmentation purposes)
    perturbation = np.random.uniform(-25, 25, size=(4, 2))  # Random perturbation for the points
    perturbed_landmarks = target_landmarks + perturbation

    # Calculate perspective transformation matrix (3x3)
    persp_matrix = cv2.getPerspectiveTransform(np.float32(perturbed_landmarks), np.float32(landmark))

    # Apply perspective transformation to the image using cv2.warpPerspective
    warped_image_np = cv2.warpPerspective(im_np, persp_matrix, (W, H))
    landmarks_homogeneous = np.hstack([landmark, np.ones((landmark.shape[0], 1))])  # Convert to homogeneous coordinates
    transformed_landmarks = (persp_matrix @ landmarks_homogeneous.T).T
    transformed_landmarks /= transformed_landmarks[:, [2]]
    transformed_landmarks = transformed_landmarks[:, :2]
    transformed_landmarks[:, 0] = np.clip(transformed_landmarks[:, 0], 0, W)
    transformed_landmarks[:, 1] = np.clip(transformed_landmarks[:, 1], 0, H)

    # Normalization điểm nguồn và điểm đích về không gian [-1, 1]
    src_point_normalized = (target_landmarks / np.array([W, H]) * 2) - 1
    des_point_normalized = (transformed_landmarks / np.array([W, H]) * 2) - 1

    # Thêm 1 để tạo tọa độ đồng nhất (homogeneous coordinates)
    src_point_augmented = np.hstack([src_point_normalized, np.ones((src_point_normalized.shape[0], 1))])

    # Tính toán ma trận affine bằng phương pháp bình phương tối thiểu
    A, _, _, _ = np.linalg.lstsq(src_point_augmented, des_point_normalized, rcond=None)

    # Ma trận affine cần được chuyển đổi từ 3x2 thành 2x3 cho PyTorch
    affine_matrix = A.T
    affine_matrix = np.array(affine_matrix, dtype = np.float32)

    # # Extract the 2D coordinates

    # # Normalize by the third homogeneous coordinate

    # 

    # # Clip the landmarks to the image boundaries

    # # Save the warped image for verification (optional)
    # print(warped_image_np.shape)
    # warped_image_save = np.array(warped_image_np * 255, dtype=np.uint8)

    # cv2.imwrite('warped_sample.jpg', warped_image_save)
    # warped_image_save = cv2.cvtColor(warped_image_save,cv2.COLOR_GRAY2BGR)
    # new_image = Image.fromarray(warped_image_save)
    # draw = ImageDraw.Draw(new_image)
    # for x, y in transformed_landmarks:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 255))  # Red points
    # for x, y in target_landmarks:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))  # Red points
    # for x, y in landmark:
    #     radius = 5
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))  # Red points
    # new_image.save('sample_2.jpg')

    # assert False
    # Return the transformed image and perspective matrix

    return torch.tensor([warped_image_np]), torch.tensor(affine_matrix, requires_grad=True)

if __name__ == "__main__":
    import cv2
    import numpy as np
    import os
    root = "/home1/data/congvu/RetinaFace/STN_dataset/dataset_1/images"
    os.makedirs('TMP', exist_ok = True)
    index = 0
    for img_path in os.listdir(root):
        index += 1
        image = Image.open(f'{root}/{img_path}')
        image = resize_with_padding(image)
        image = np.array(image.convert("L"))/255.
        mean = np.array((0.5))
        image = (image - mean)/ mean
        image = np.expand_dims(image, axis = -1)
        image = image.transpose((2,0,1))
        
        image = torch.tensor([image])
        image, recovered_image = Affine_aug(image)
        image = image[0].numpy().transpose((1,2,0))
        image = (image*mean+mean)*255
        image = np.array(image, dtype = 'uint8')
        cv2.imwrite(f'TMP/{index}.png',image)    
        index += 1
