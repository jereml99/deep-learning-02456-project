# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:06:45 2023

@author: Ila
"""


"""
ToDo: 
    - is the rotation part needed? 
        - if so, figure out a way for resizing as it needes to be done afterwards. 
          (think about the inputs of the img_resizer)
"""


import glob
import os
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow


from DataHandler import DataHandler
data_handler = DataHandler()


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    
    

""" read data """
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
data_handler.get_img_sizes(data)


""" image rotation """ 
rotated_data = data_handler.img_rotater(data)

# show the result from a random image
rand_idx = random.randint(0, len(rotated_data))
show_comparison(data[rand_idx], rotated_data[rand_idx], 'rotated')

# re-check the image sizes
data_handler.get_img_sizes(rotated_data)


""" resize image """  
new_height = 360
new_width = 640
resized_data = data_handler.img_resizer(rotated_data, new_height, new_width)

# re-check the image sizes
data_handler.get_img_sizes(resized_data)























