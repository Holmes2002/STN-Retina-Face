import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


def get_Perspective(label, height, width, input_image, visualize = False):

# Source points from the annotation array
    if visualize:
        source_points_int = np.int32([
            [int(label[4]), int(label[5])],   # l0_x, l0_y (top-right)
            [int(label[7]), int(label[8])],   # l1_x, l1_y (top-left)
            [int(label[13]), int(label[14])], # l3_x, l3_y (bottom-right)
            [int(label[16]), int(label[17])]  # l4_x, l4_y (bottom-left)
        ])

        # Draw circles on each of the corners
        colors = [(0, 0, 255),   # Red
                  (255, 0, 0),   # Blue
                  (0, 255, 0),   # Green
                  (0, 0, 0)]     # Black

        # Draw circles on each of the corners with different colors
        # for i, point in enumerate(source_points_int):
        #     cv2.circle(input_image, (point[0], point[1]), radius=5, color=colors[i], thickness=-1)
        # cv2.imwrite("4_corner.jpg", input_image)
    
    source_points = np.float32([
        [label[7], label[8]],   # l0_x, l0_y (top-right)
        [label[4], label[5]],   # l1_x, l1_y (top-left)
        [label[16], label[17]], # l3_x, l3_y (bottom-right)
        [label[13], label[14]], # l3_x, l3_y (bottom-right)
        # [label[16], label[17]]  # l4_x, l4_y (bottom-left)
    ])

    # Define the destination points (e.g., a rectangle)
    # These are the corners of the output image you want to map to
    dest_points = np.float32([
        [width, 0],        # top-right
        [0, 0],            # top-left
        [width, height],   # bottom-right
        [0, height]        # bottom-left
    ])

    # Get the perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
    if visualize:
        transformed_image = cv2.warpPerspective(input_image, perspective_matrix, (width, height))
        cv2.imwrite('output_image.jpg', transformed_image)
        return transformed_image
    perspective_matrix = perspective_matrix.flatten().reshape(-1)
    perspective_matrix = perspective_matrix[:8]

    return perspective_matrix
class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

        os.makedirs('STN_dataset', exist_ok = True)
        os.makedirs('STN_dataset/labels', exist_ok = True)
        os.makedirs('STN_dataset/images', exist_ok = True)

        index_images = 0
        # for index in range(len(self.imgs_path)):
        #     img = cv2.imread(self.imgs_path[index])
        #     name = self.imgs_path[index].split('/')[-1].replace('jpg','txt')
        #     # file = open(f"STN_dataset/{name}", 'w')
        #     height, width, _ = img.shape

        #     labels = self.words[index]
        #     annotations = np.zeros((0, 15))
        #     if len(labels) == 0:
        #         return annotations
        #     for idx, label in enumerate(labels):
        #         try:
        #             index_images += 1
        #             annotation = np.zeros((1, 15))
        #             # bbox
        #             annotation[0, 0] = label[0]  # x1
        #             annotation[0, 1] = label[1]  # y1
        #             annotation[0, 2] = label[0] + label[2]  # x2
        #             annotation[0, 3] = label[1] + label[3]  # y2
        #             crop_image = img[int(label[1]): int(label[1]+label[3]), int(label[0]): int(label[0]+label[2]), :]
        #             height, width, _ = crop_image.shape
        #             transformed_image = get_Perspective(label, height, width, img, True)
        #             cv2.imwrite(f"STN_dataset/images/{index_images}.jpg", crop_image)
        #             cv2.imwrite(f"STN_dataset/labels/{index_images}.jpg", transformed_image)
        #             index_images += 1
        #         except: 
        #             continue
        #         line = f'LP 0.00 0 0.00 {label[0]} {label[1]} {label[2]+label[0]} {label[1]+label[3]} 0.00 0.00 0.00 0.00 0.00 0.00 0.00\n'
        #         file.write(line)
            # file_images.write(f"{self.imgs_path[index]}\n")


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            #### Matrix Perspective
            # crop_image = img[int(label[1]): int(label[1]+label[3]), int(label[0]): int(label[0]+label[2]), :]
            # height, width, _ = crop_image.shape
            # perspective_matrix = get_Perspective(label, height, width, img)
            # assert False


            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
            # perspective_matrix = np.expand_dims(perspective_matrix, 0)
        
        # if True:
        #     new_target = np.concatenate((target[0, :4], perspective_matrix[0,:], target[0, -1:]   ))
        #     new_target = np.expand_dims(new_target, 0)
        #     return torch.from_numpy(img), new_target
        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

if __name__ == "__main__":
    file_boxes = open('boxes.txt', 'w')
    file_images = open('images.txt', 'w')
    data = WiderFaceDetection("licenseplate/train/label.txt")
