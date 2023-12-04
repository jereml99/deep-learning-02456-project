import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.deeplab import Deeplab
import glob
from dataset import CustomDataset
from torch.utils.data import DataLoader

from test_utils import test_model
from models.unet import UNet


SAMPLES_DIR = "data_two"

def train_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/train/**/*.npy")
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

def validation_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/val/**/*.npy")
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=True,  drop_last=True)

def test_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/test/**/*.npy")
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=False,  drop_last=True)

def main():
    wandb_logger = WandbLogger(project='car-segmentation')

    model = UNet(in_channels=3, out_channels=10)
    trainer = pl.Trainer(max_epochs=20, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, train_dataloader(), validation_dataloader())


    # Run testing after training
    mean_iou = test_model(model, test_dataloader())
    
    print(f"Mean IoU on the test set: {mean_iou}")
    wandb.finish()

if __name__ == "__main__":
    main()
