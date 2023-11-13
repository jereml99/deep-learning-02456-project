# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:32:10 2023

@author: ila
"""

import torch
import torchvision
from dataset import DeloitteDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("= Saving checkpoint")
    torch.save(state, filename)
    
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    








































