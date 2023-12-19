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
    
    

def get_loaders(train_dir, 
                train_maskdir,
                val_dir,
                val_maskdir,
                batch_size, 
                train_transform,
                val_transform,
                num_workers=4,
                pin_memory=True):
    
    # create a Deloitte dataset
    train_ds = DeloitteDataset(image_dir=train_dir, transfrom=train_transform)
    # provide batches to the training process
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)
    # create a validation dataset 
    val_ds = DeloitteDataset(image_dir=val_dir, transfrom=val_transform)
    # provide batches to the validation process
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)
    return train_loader, val_loader



def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # binary classification
            #preds = torch.sigmoid(model(x))
            #preds = (preds > 0.5).float()
            #num_correct += (preds == y).sum()
            
            # multi-class classification
            _, preds = torch.max(model(x), 1)
            num_correct += (preds == y).sum().item()
            
            num_pixels += torch.numel(preds)
            
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    
    
    
def save_predictions_as_imgs(loader, 
                             model,
                             folder="saved_images/",
                             device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            # binary classification
            #preds = torch.sigmoid(model(x))
            #preds = (preds > 0.5).float()
            
            # multi-class classification
            _, preds = torch.max(model(x), 1)
            
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/true_{idx}.png")
        
    model.train()






































