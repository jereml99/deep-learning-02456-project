# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:06:45 2023

@author: Ila
"""

import glob
import os
import io
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from skimage.transform import resize


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()


""" Data Prep """
# read a small subset of the data
in_dir = 'labeled_data_sample/'
types = ('*.png', '*.jpg')
all_images = []
for img_type in types:
    all_images.extend(glob.glob(in_dir + img_type))
    
data = []
# store the data in a list 
for img_idx in all_images:
    name_no_ext = os.path.splitext(img_idx)[0]
    name_ext = os.path.splitext(img_idx)[1]

    img = imread(f"{name_no_ext}{name_ext}")
    data.append(img)

# show a random image from the loaded data
rdn = random.randint(0, len(data))
plt.figure()
plt.title('random image from the loaded data')
imshow(data[rdn])

# check the image sizes 
image_shapes = []
for img in data:
    img_size = [img.shape[0], img.shape[1]]
    if img_size not in image_shapes:
        image_shapes.append(img_size)
print(F"available image shapes: {image_shapes}")


# image re-sizing  
def img_resizer(img_list, new_height, new_width):
    resized_imgs = []
    for img in img_list:
        image_new_size = (new_height, new_width, 3)
        resized_img = resize(img, output_shape=image_new_size, mode='reflect', anti_aliasing=True)
        resized_imgs.append(resized_img)
    return resized_imgs

new_height = 360
new_width = 640
resized_data = img_resizer(data, new_height, new_width)

# re-check the image sizes 
new_image_shapes = []
for img in resized_data:
    img_size = [img.shape[0], img.shape[1]]
    if img_size not in new_image_shapes:
        new_image_shapes.append(img_size)
print(F"new available image shapes: {new_image_shapes}")























