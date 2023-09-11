import cv2
import numpy as np

def adjust_image_for_night(image, darkness_factor=0.5):
    dark_image = np.clip(image * darkness_factor, 0, 255).astype(np.uint8)
    return dark_image

def apply_image_modifications(image_paths, darkness_factor=0.5):
    modified_images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        dark_image = adjust_image_for_night(image, darkness_factor)
        modified_images.append(dark_image)
        cv2.imwrite(image_path, dark_image)

    return modified_images