import os
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import random
from torchvision import transforms
from torchvision.transforms import functional as F
def random_transform_image(image, crop_size=(224, 224), zoom_range=(0.8, 1.2), rotation_range=(-30, 30)):
    # Randomly select crop position
    transformed_image = image.resize((256,256))
    width, height = image.size
    
    # Apply random crop
    if random.random()<0.4:
        left = random.randint(0, width - crop_size[0])
        top = random.randint(0, height - crop_size[1])
        right = left + crop_size[0]
        bottom = top + crop_size[1]

        transformed_image = image.crop((left, top, right, bottom))
    if random.random()<0.4:
        # Randomly select zoom factor
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        new_size = (int(crop_size[0] * zoom_factor), int(crop_size[1] * zoom_factor))
        
        # Apply zoom in/out
        transformed_image = transformed_image.resize(new_size, Image.BILINEAR)
    transformed_image = transformed_image.resize((256,256))
    
    return transformed_image

def transform_JPEGcompression(image, label, compress_range = (30, 100)):
    """
    Perform random JPEG Compression on the image and label.
    """
    assert compress_range[0] < compress_range[1], "Invalid compression range: {} vs {}".format(compress_range[0], compress_range[1])

    jpegcompress_value = random.randint(compress_range[0], compress_range[1])

    # Handle JPEG compression for the image
    out_image = BytesIO()
    image.save(out_image, 'JPEG', quality=jpegcompress_value)
    out_image.seek(0)  # Go back to the start of the BytesIO buffer
    rgb_image = Image.open(out_image)

    # Handle JPEG compression for the label
    out_label = BytesIO()
    label.save(out_label, 'JPEG', quality=jpegcompress_value)
    out_label.seek(0)
    compressed_label = Image.open(out_label)

    return rgb_image, compressed_label


def transform_gaussian_noise(img, label, mean = 0.0, var = 10.0):
    '''
        Perform random gaussian noise
    '''
    img = np.array(img)
    label = np.array(label)
    height, width, channels = img.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma,(height, width, channels))
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy = noisy.astype(np.uint8)

    noisy_label = label + gauss
    cv2.normalize(noisy_label, noisy_label, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_label = noisy_label.astype(np.uint8)
    return Image.fromarray(noisy), Image.fromarray(noisy_label)


def _motion_blur(img, label):
    # Ensure the input images are in NumPy array format
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    # Specify the kernel size
    kernel_size = random.randint(3, 7)

    # Create the vertical kernel
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)  # Vertical line

    # Create the horizontal kernel
    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)  # Horizontal line

    # Normalize the kernels
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply motion blur based on random choice
    if np.random.uniform() > 0.5:
        # Apply vertical blur
        blurred = cv2.filter2D(img, -1, kernel_v)
        blurred_label = cv2.filter2D(label, -1, kernel_v)
    else:
        # Apply horizontal blur
        blurred = cv2.filter2D(img, -1, kernel_h)
        blurred_label = cv2.filter2D(label, -1, kernel_h)
    return Image.fromarray(blurred), Image.fromarray(blurred_label)
    # return blurred, blurred_label

def _unsharp_mask(image, kernel_size=5, sigma=-.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def _increase_contrast(img, kernel_size):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=np.random.uniform(0.001, 4.0), tileGridSize=(kernel_size,kernel_size))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def transform_random_blur(img):
    img = np.array(img)
    flag = np.random.uniform()
    kernel_size = random.choice([ 17,19,21,23,25,27])
    if flag >= 0.6:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), np.random.uniform(0.0, 2.0))
    elif flag >= 0.3:
        kernel_size = random.choice([ 21,23,25,27,29,31])
        img = _motion_blur(img, kernel_size)
    elif flag >= 0.0:
        kernel_size = random.choice([   17,19,21,23, 25,27])
        img = cv2.blur(img, (kernel_size, kernel_size))
    return Image.fromarray(img)
def transform_random_blur_test(img, label):
    img = np.array(img)
    label = np.array(label)
    flag = np.random.uniform()
    kernel_size = random.choice([3, 5, 7, 9, 11])
    
    if flag > 0.5:
        kernel_size = 17
        img = _unsharp_mask(img, kernel_size = kernel_size)
        label = _unsharp_mask(label, kernel_size = kernel_size)
    elif flag >= 0.0:
        img = _increase_contrast(img, kernel_size)
        label = _increase_contrast(label, kernel_size)
    return Image.fromarray(img), Image.fromarray(label)

def transform_adjust_gamma(image, lower = 0.2, upper = 2.0):
    image = np.array(image)
    gamma = np.random.uniform(lower, upper)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return Image.fromarray(cv2.LUT(image, table))

# def transform_blur(img):
#     flag = np.random.uniform()
#     kernel_size = random.choice([3, 5, 7, 9])
#     img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
#     return img

def transform_to_gray(img):
    '''
        Perform random gaussian noise
    '''
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(gray)

def transform_resize(image, resize_range = (24, 112), target_size = 112):
    assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
    resize_value = random.randint(resize_range[0], resize_range[1])
    resize_image = image.resize((resize_value, resize_value))
    return resize_image.resize((target_size, target_size))


# def transform_eraser(image):
#     if np.random.uniform() < 0.1:
#         mask_range = random.randint(0, 3)
#         image_array = np.array(image, dtype=np.uint8)
#         image_array[(7-mask_range)*16:, :, :] = 0
#         return Image.fromarray(image_array)
#     else:
#         return image

def transform_color_jiter(sample, label, brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1):
    photometric = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            photometric.get_params(photometric.brightness, photometric.contrast,
                                                  photometric.saturation, photometric.hue)
    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            sample = F.adjust_brightness(sample, brightness_factor)
            label = F.adjust_brightness(label, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            sample = F.adjust_contrast(sample, contrast_factor)
            label = F.adjust_contrast(label, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            sample = F.adjust_saturation(sample, saturation_factor)
            label = F.adjust_saturation(label, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            sample = F.adjust_hue(sample, hue_factor)
            label = F.adjust_hue(label, hue_factor)
    return sample, label

def random_augment(sample):
    # Input is RGB image

    # Blur augmentation
    # if np.random.uniform() < 0.7:
    if np.random.uniform() < 0.9:
        sample = transform_random_blur(sample)

    return sample
def random_augment_test(sample, label):
    # Input is RGB image

    # Blur augmentation
    # if np.random.uniform() < 0.7:
    if np.random.uniform() < 0.2:
        sample, label = transform_random_blur_test(sample, label)
    elif np.random.uniform() < 0.4:
        sample, label = transform_JPEGcompression(sample, label)
    elif np.random.uniform() < 0.6:
        sample, label = transform_color_jiter(sample, label)
    elif np.random.uniform() < 0.8:
        sample, label = transform_gaussian_noise(sample, label)
    else:
        sample, label = _motion_blur(sample, label)
    return sample, label

if __name__ == '__main__':
    import glob
    from PIL import Image
    input_path = '/home1/data/congvu/TMP/cropped_LPs/lp_100.png'
    img = Image.open(input_path).convert("RGB")
    aug_img, _ = random_augment_test(img, img)
    print(aug_img)
    aug_img.save('aug_sample.png')




