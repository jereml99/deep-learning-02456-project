# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:51:02 2023

@author: ila
"""

import os
from PIL import Image 
from torch.utils.data import Dataset
import numpy as np


"""
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
image_dir = os.path.join(parent_dir, 'data_samples\Train')
"""
class DeloitteDataset(Dataset):
    
    def __init__(self, image_dir, transfrom=None):
        self.image_dir = image_dir
        self.transfrom = transfrom
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img_file = np.load(img_path)
        image = img_file[:, :, :3]
        mask = img_file[:, :, 3]
        
        if self.transfrom is not None:
            augmentations = self.transfrom(image=image, mask=mask)
            
        return image, mask 
            












































