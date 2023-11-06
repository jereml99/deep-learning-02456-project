# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:30:08 2023

@author: Ila
"""

import random
from skimage.transform import resize
from skimage.transform import rotate


class DataHandler():
    
    def __init__(self):
        pass 
    
    
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
            rotated_img = rotate(img_org, rotation_angle, resize=True, mode="reflect")
            rotated_imgs.append(rotated_img)
            return rotated_imgs


    def img_resizer(self, data, new_height, new_width):
        resized_imgs = []
        for img in data:
            image_new_size = (new_height, new_width, 3)
            resized_img = resize(img, output_shape=image_new_size, mode='reflect', anti_aliasing=True)
            resized_imgs.append(resized_img)
        return resized_imgs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    