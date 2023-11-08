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
import numpy as np
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
    plt.show()
    

""" read data """
# read a small subset of the data
in_dir = 'data_samples/*.npy'
npy_files = glob.glob(in_dir)

    
data = []
# store the data in a list 
for file_name in npy_files:
    array = np.load(file_name)
    data.append(array)

# show a random image from the loaded data
rdn = random.randint(0, len(data)-1)
plt.figure()
plt.title('random image from the loaded data')
imshow(data[rdn])
plt.show()


# check the image sizes 
data_handler.get_img_sizes(data)
data_handler.show_image_with_classes(data[rdn])

""" image rotation """ 
rotated_data = data_handler.img_rotater(data)

# show the result from a random image
rand_idx = random.randint(0, len(rotated_data)-1)
show_comparison(data[rand_idx], rotated_data[rand_idx], 'rotated')

# re-check the image sizes
data_handler.get_img_sizes(rotated_data)


""" resize image """  
new_height = 360
new_width = 640
resized_data = data_handler.img_resizer(rotated_data, new_height, new_width)

show_comparison(data[rand_idx], resized_data[rand_idx], 'resized')
data_handler.get_img_sizes(resized_data)
























