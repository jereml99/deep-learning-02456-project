# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:30:08 2023

@author: Ila
"""

import random
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize
from skimage.transform import rotate
from matplotlib.pyplot import imshow

class DataHandler():
    
    def __init__(self):
        self.class_to_color = {
            10: [255, 165, 0],    # orange
            20: [0, 100, 0],      # dark green
            30: [255, 255, 0],    # yellow
            40: [0, 255, 255],    # cyan
            50: [128, 0, 128],    # purple
            60: [144, 238, 144],  # light green
            70: [0, 0, 255],      # blue
            80: [255, 192, 203],  # pink
        } 
            
    
    def get_img_sizes(self, data):
        image_shapes = []
        for img in data:
            img_size = [img.shape[0], img.shape[1]]
            if img_size not in image_shapes:
                image_shapes.append(img_size)
        return print(F"new available image shapes: {image_shapes}")


    def img_rotater(self, data):
        rotated_imgs = []
        for img_org in data:
            rotation_angle = random.randint(0, 360)
            rotated_img = rotate(img_org, rotation_angle, resize=True, mode="edge")
            rotated_imgs.append(rotated_img)
            return rotated_imgs


    def img_resizer(self, data, new_height, new_width):
        resized_imgs = []
        for img in data:
            image_new_size = (new_height, new_width, 4)
            resized_img = resize(img, output_shape=image_new_size, mode='edge')
            resized_imgs.append(resized_img)
        return resized_imgs
    
    def show_image_with_classes(self, img, ax: plt.Axes=None):
        colored_image = self.apply_class_colors(img)
        if ax is None:
            plt.figure()
            plt.title('Image with Applied Class Colors')
            imshow(colored_image)
            plt.axis('off')  # Hide the axis
            plt.show()
        else:
            ax.set_title('Image with Applied Class Colors')
            ax.imshow(colored_image)
            ax.axis('off')  # Hide the axis

    def apply_class_colors(self, img):
        colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for class_value, color in self.class_to_color.items():
            mask = img[:, :, 3] == class_value
            colored_img[mask] = color
        return colored_img
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    