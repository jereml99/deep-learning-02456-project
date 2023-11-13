# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:08:03 2023

@author: ila
"""
# https://www.youtube.com/watch?v=IHq1t7NxS8k&t=3100s

import os
import torch
from torchvision import transforms # import albumentaions as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim
from model import UNet
#from utils import (
#    load_checkpoints, 
#    save_checkpoints,
#    get_loaders,
#    check_accuracy, 
#    save_predictions_as_imgs)


""" Hyperparametrs etc """
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = os.path.join(parent_dir, 'data_samples\Train')
VAL_IMG_DIR = os.path.join(parent_dir, 'data_samples\Validation')



def train_fn(loader, model, optimizer, loss_fn, scaler):
    """ tain one epoch """
    
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(deviece=DEVICE)
        
    # forward 
    with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
    # backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # update tqdm loop
    loop.set_postfix(loss=loss.item())
    
    


def main():
    """
    train_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                 A.Rotate(limit=35),
                                 A.HorizontalFlip(p=0.5),
                                 A.VerticalFlip(p=0.1), 
                                 A.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[1.0, 1.0, 1.0],
                                             max_pixel_value = 255.0),
                                 ToTensorV2()])
    
    val_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                               A.Normalize(mean=[0.0, 0.0, 0.0],
                                           std=[1.0, 1.0, 1.0],
                                           max_pixel_value = 255.0),
                               ToTensorV2()])
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomRotation(35),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])


    # update the outchannels - how many classes do we have?
    model = UNet(in_channels=3, out_channels=5).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        VAL_IMG_DIR,
        BATCH_SIZE, 
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
        )
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save model
        # check accuracy 
        # print some examples to a folder
        
        

if __name__ == "__main__":
    main()
































