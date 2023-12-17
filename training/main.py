import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.deeplab import Deeplab
from models.deeplab_restnet101 import Deeplab_RestNet_101
import glob
from dataset import CustomDataset
from torch.utils.data import DataLoader

from test_utils import test_model
from models.unet import UNet


SAMPLES_DIR = "data_samples"

def train_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/train/**/*.npy", recursive=True)
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

def validation_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/val/**/*.npy", recursive=True)
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=16, shuffle=True,  drop_last=True)

def test_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/test/**/*.npy", recursive=True)
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=False,  drop_last=True)


models = {
    "Deeplab_restnet_50" : Deeplab_RestNet_101(num_classes=9),
    "Unet": UNet(in_channels=3, out_channels=10),
    "Deeplab_restnet_101": Deeplab_RestNet_101(num_classes=9)
}

def main():
    wandb_logger = WandbLogger(project='car-segmentation')

    model = models["Deeplab_restnet_101"]
    trainer = pl.Trainer(max_epochs=20, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, train_dataloader(), validation_dataloader())


    # Run testing after training
    mean_iou = test_model(model, test_dataloader())
    
    print(f"Mean IoU on the test set: {mean_iou}")
    wandb.finish()

if __name__ == "__main__":
    main()
