import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, numpy_files):
        self.numpy_files = numpy_files

    def __len__(self):
        return len(self.numpy_files)

    def __getitem__(self, idx):
        data = np.load(self.numpy_files[idx])
        image = data[:, :, :3]
        mask = (data[:, :, 3]/10) # Change classes to be from 0 to 9 instead of 0 to 90 

        image = image / 255.0
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
