import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.deeplab import Deeplab
import glob
from dataset import CustomDataset
from torch.utils.data import DataLoader

SAMPLES_DIR = "data_samples"

def train_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/Train/*.npy")
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=True)

def validation_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/Validation/*.npy")
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=True)

def test_dataloader():
    samples = glob.glob(SAMPLES_DIR+"/Test/*.npy")
    dataset = CustomDataset(numpy_files=samples)
    return DataLoader(dataset, batch_size=8, shuffle=True)

def main():
    wandb_logger = WandbLogger(project='car-segmentation')

    model = Deeplab(num_classes=9)
    trainer = pl.Trainer(max_epochs=2, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, train_dataloader(), validation_dataloader())

    wandb.finish()

if __name__ == "__main__":
    main()
