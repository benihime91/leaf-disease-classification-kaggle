from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import models
import albumentations as A

from datasets import PlantDataset


class LitModel(pl.LightningModule):
    def __init__(self,
                 output_dims: int,
                 learning_rate: float,
                 weight_decay: float,
                 total_steps: int,
                 class_weights: Optional[torch.Tensor] = None,
                 hidden_dims: int = 512,
                 ):

        super().__init__()
        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.total_steps   = total_steps
        self.output_dims   = output_dims

        # Define PyTorch model
        self.classifier = models.resnext50_32x4d(pretrained=True, progress=True)

        # dims of outputs of the classifier
        self.base_output_dims = self.classifier.fc.out_features

        self.hidden_layers = nn.Sequential(
            nn.BatchNorm1d(base_output_dims),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(base_output_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, self.output_dims)
        )

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def freeze_classifier(self):
        for params in self.classifier.parameters():
            params.requires_grad = False

    def unfreeze_classifier(self):
        for params in self.classifier.parameters():
            params.requires_grad = True


    def forward(self, x):
        output = self.classifier(x)
        output = self.hidden_layers(output)
        return output


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
        # init AdamW optimizer and OneCycleLR Scheduler
        model  = nn.Sequential(self.classifier, self.hidden_layers)
        params = [p for p in model.parameters() if p.requires_grad]
        opt    = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        sch = torch.optim.lr_scheduler.OneCycleLR(opt, total_steps=self.total_steps, max_lr=self.learning_rate)
        sch = {"scheduler": sch, "interval": "step", "frequency": 1, }
        return [opt], [sch]


class LitDatatModule(pl.LightningDataModule):
    def __init__(self,
                 df_train: pd.DataFrame,
                 df_valid: pd.DataFrame,
                 df_test: pd.DataFrame,
                 batch_size: int,
                 transforms: A.Compose
                 ):

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
