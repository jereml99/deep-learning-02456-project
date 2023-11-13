# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:02:36 2023

@author: ila
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from unet_model import UNet


""" sanity checking """


""" select a sample """
def get_sample ():
    sample_name = 'black_5_doors_0027.npy'

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data_samples')
    train_dir = os.path.join(data_dir, 'Train')
    sample_data_dir = os.path.join(train_dir, sample_name)
    
    return sample_data_dir

sample_data_dir = get_sample()
data = np.load(sample_data_dir, mmap_mode='r') #numpy array
print(data.shape)

# image + mask prep
image = data[:, :, :3]
mask = (data[:, :, 3]/10) # Change classes to be from 0 to 9 instead of 0 to 90 
image = image / 255.0
image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
mask_tensor = torch.from_numpy(mask).long()


""" define network """ 
# Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
net = UNet(n_channels=3, n_classes=10)
print(net)


""" Train network """
net.train()
output = net()






















