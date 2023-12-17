from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50



class Deeplab(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = deeplabv3_resnet50(num_classes=num_classes+1) # Adding background that is not included in the class number

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks)
        self.log('validation_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer