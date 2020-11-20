import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import models

from datasets import PlantDataset


class LitModel(pl.LightningModule):
    def __init__(self, output_dims, learning_rate, weight_decay, total_steps, class_weights=None):
        super().__init__()
        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.output_dims = output_dims

        classifier = models.resnet50(pretrained=True, progress=True)
        # dims of outputs of the classifier
        base_output_dims = self.classifier.fc.out_features
        # unfreeze the classifier
        classifier.requires_grad_(True)

        lin_1 = nn.Sequential(
            nn.BatchNorm1d(base_output_dims),
            nn.Dropout(0.25),
            nn.ReLU(),
        )

        lin_2 = nn.Sequential(
            nn.Linear(base_output_dims, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.ReLU(),
        )

        lin_3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            classifier,
            lin_1,
            lin_2,
            lin_3,
            nn.Linear(512, self.output_dims)
        )

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, total_steps=self.total_steps, max_lr=self.learning_rate)
        sch = {"scheduler": sch, "interval": "step", "frequency": 1, }
        return [opt], [sch]


class LitDatatModule(pl.LightningDataModule):
    def __init__(self, df_train, df_valid, df_test, batch_size, transforms):
        super().__init__()
        # Set our init args as class attributes
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.batch_size = batch_size
        self.transforms = transforms

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train = PlantDataset(self.df_train, self.transforms["train"])
            self.valid = PlantDataset(self.df_valid, self.transforms["valid"])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.df_test is not None:
                self.test = PlantDataset(self.df_test, self.transforms["test"])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)
