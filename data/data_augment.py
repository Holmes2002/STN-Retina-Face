import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof
import os

def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])


	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)
def get_Perspective(label, height, width, input_image, visualize = False):

# Source points from the annotation array
    input_image = input_image.astype('uint8')
    source_points = np.float32([
        [label[2], label[3]],   # l0_x, l0_y (top-right)
        [label[0], label[1]],   # l1_x, l1_y (top-left)
        [label[8], label[9]], # l3_x, l3_y (bottom-right)
        [label[6], label[7]]  # l4_x, l4_y (bottom-left)
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
        os.makedirs('check_landmark', exist_ok = True)
        source_points_int = np.int32([
            [label[8], label[9]],   # l0_x, l0_y (top-right)
            [label[6], label[7]],   # l1_x, l1_y (top-left)
            [label[2], label[3]], # l3_x, l3_y (bottom-right)
            [label[0], label[1]],
            # [label[0], label[1]]
              # l4_x, l4_y (bottom-left)
        ])

        # Draw circles on each of the corners
        colors = [(0, 0, 255),   # Red
                  (255, 0, 0),   # Blue
                  (0, 255, 0),   # Green
                  (0, 0, 0)]     # Black

        # Draw circles on each of the corners with different colors
        for i, point in enumerate(source_points_int):
            cv2.circle(input_image, (point[0], point[1]), radius=5, color=colors[i], thickness=-1)
        index_image = len(os.listdir('check_landmark')) +1
        # cv2.imwrite("4_corner.jpg", input_image)
        transformed_image = cv2.warpPerspective(input_image, perspective_matrix, (width, height))
        cv2.imwrite(f'check_landmark/{index_image}.jpg', transformed_image)

    perspective_matrix = perspective_matrix.flatten().reshape(-1)
    perspective_matrix = perspective_matrix[:8]

    return (perspective_matrix)


# class preproc(object):

#     def __init__(self, img_dim, rgb_means):
#         self.img_dim = img_dim
#         self.rgb_means = rgb_means

#     def __call__(self, image, targets):
#         assert targets.shape[0] > 0, "this image does not have gt"

#         boxes = targets[:, :4].copy()
#         labels = targets[:, -1].copy()
#         landm = targets[:, 4:-1].copy()

#         image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
#         image_t = _distort(image_t)
#         image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
#         image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
#         height, width, _ = image_t.shape
#         image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)


#         size_LP = []
#         for box in boxes_t:
#             w, h = int(box[2]-box[0]), int( box[3] - box[1])
#             size_LP.append((w,h))
#         perspective_matrixs = []
#         for index, land_markd in enumerate(landm_t):
#             target = get_Perspective(land_markd, size_LP[index][1], size_LP[index][0], image_t)
#             perspective_matrixs.append(target)
#         perspective_matrixs = np.array(perspective_matrixs)
#         # assert False
#         boxes_t[:, 0::2] /= width
#         boxes_t[:, 1::2] /= height

#         # landm_t[:, 0::2] /= width
#         # landm_t[:, 1::2] /= height
#         labels_t = np.expand_dims(labels_t, 1)
#         targets_t = {'boxes': boxes_t, 'wrap_matrix':perspective_matrixs, "labels":labels_t}
#         return image_t, targets_t

class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()


        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        

        size_LP = []
        for box in boxes_t:
            w, h = int(box[2]-box[0]), int( box[3] - box[1])
            size_LP.append((w,h))
            break
        perspective_matrixs = []
        for index, land_markd in enumerate(landm_t):
            target = get_Perspective(land_markd, size_LP[index][1], size_LP[index][0], image_t, visualize = True)
            perspective_matrixs.append(target)
            break
        perspective_matrixs = np.array(perspective_matrixs)
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        size_image = np.array([[width, height]])
        labels_t = np.expand_dims(labels_t, 1)
        landm_t = np.delete(landm_t, [4, 5], axis=1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t, perspective_matrixs))
        if np.all(perspective_matrixs == 0.):
            assert False

        return image_t, targets_t