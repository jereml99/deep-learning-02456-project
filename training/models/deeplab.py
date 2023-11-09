from matplotlib import pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import glob

from dataset import CustomDataset

SAMPLES_DIR = "data_samples"

class Deeplab(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = deeplabv3_resnet50(num_classes=num_classes+1) # Adding background that is not included in the class number

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        samples = glob.glob(SAMPLES_DIR+"/Train/*.npy")
        dataset = CustomDataset(numpy_files=samples)
        return DataLoader(dataset, batch_size=4, shuffle=True)
    
    def test_dataloader(self):
        samples = glob.glob(SAMPLES_DIR+"/Test/*.npy")
        dataset = CustomDataset(numpy_files=samples)
        return DataLoader(dataset, batch_size=4, shuffle=True)
    

    def on_train_end(self):
        # Switch to eval mode
        self.eval()
        self.freeze()

        # Get a sample batch from the validation dataset
        val_loader = self.test_dataloader()
        batch = next(iter(val_loader))
        images, masks = batch
        
        # Perform inference
        with torch.no_grad():
            outputs = self(images)
        
        # Convert outputs to predicted class indices
        _, preds = torch.max(outputs, 1)

        # Plot the results
        self.plot_sample(images[0], masks[0], preds[0])
        
        # Switch back to train mode
        self.unfreeze()
        self.train()

    def plot_sample(self, image, mask, pred):
        # Assuming image, mask, and pred are PyTorch tensors
        # Normalize image for visualization if necessary
        image = image.cpu().numpy().transpose((1, 2, 0))
        mask = mask.cpu().numpy()
        pred = pred.cpu().numpy()

        # Create subplots
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display the image
        ax[0].imshow(image)
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        # Display the ground truth mask
        ax[1].imshow(mask)
        ax[1].set_title('Ground Truth Mask')
        ax[1].axis('off')

        # Display the predicted mask
        ax[2].imshow(pred)
        ax[2].set_title('Predicted Mask')
        ax[2].axis('off')

        plt.show()
