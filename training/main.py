import pytorch_lightning as pl
from models.deeplab import Deeplab


def main():
    model = Deeplab(num_classes=9)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model)

if __name__ == "__main__":
    main()