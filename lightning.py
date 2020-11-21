from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import models
import albumentations as A

from datasets import PlantDataset


# -----------------------------------
# LIGHTNING MODULE :
# ------------------------------------

class LightningModel_resnext50_32x4d(pl.LightningModule):
    def __init__(self,
                 output_dims: int,
                 learning_rate: float,
                 weight_decay: float,
                 total_steps: int,
                 class_weights: Optional[torch.Tensor] = None,
                 hidden_dims: int = 512,
                 ):
        """
        Lightning Module with resnext50_32x4d backbone, AdamW optimizer & CosineAnnealingWarmRestarts

        Args:
            1. output_dims: number of output classes
            2. learning_rate: learning_rate for AdamW optimizer
            3. weight_decay: weight_decay for AdamW optimizer
            4. total_steps: number of total steps to train for 
            5. class_weights: a tensor containing the weights for nn.CrossEntropyLoss
            6. hidden_dims: number of nodes of the hidden layer connecting the base and output
        """

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
            nn.BatchNorm1d(self.base_output_dims),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_output_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, self.output_dims)
        )

        # model
        self.net = nn.Sequential(self.classifier, self.hidden_layers)

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        

    def configure_optimizers(self):
        opt_fn = torch.optim.AdamW
        sch_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

        # init AdamW optimizer and OneCycleLR Scheduler
        params = [p for p in self.net.parameters() if p.requires_grad]
        
        opt = opt_fn(params, lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.9, 0.99))
        
        sch = sch_fn(opt, T_0=10, T_mult=2, last_epoch=-1, eta_min=1e-07)

        sch = {"scheduler":sch, "interval": "step", "frequency":1}
        
        return [opt], [sch]

    def freeze_classifier(self):
        for params in self.classifier.parameters():
            params.requires_grad = False
        self.net = nn.Sequential(self.classifier, self.hidden_layers)

    def unfreeze_classifier(self):
        for params in self.classifier.parameters():
            params.requires_grad = True
        self.net = nn.Sequential(self.classifier, self.hidden_layers)


    def forward(self, x):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        # 2. Classifier (returns logits):
        return self.net(x)


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
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        acc = accuracy(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)




# -----------------------------------
# LIGHTNING DATA MODULE :
# ------------------------------------
class LitDatatModule(pl.LightningDataModule):
    def __init__(self,
                 df_train: pd.DataFrame,
                 df_valid: pd.DataFrame,
                 df_test: pd.DataFrame,
                 batch_size: int,
                 transforms: A.Compose,
                 pin_memory: bool = True,
                 num_workers: int = 0,
                 ):

        super().__init__()
        # Set our init args as class attributes
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.batch_size = batch_size
        self.transforms = transforms
        self.pin_memory = pin_memory
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train = PlantDataset(self.df_train, self.transforms["train"])
            self.valid = PlantDataset(self.df_valid, self.transforms["valid"])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = PlantDataset(self.df_test, self.transforms["test"])

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train, 
            shuffle=True, 
            batch_size=self.batch_size, 
            pin_memory=self.pin_memory, 
            num_workers=self.num_workers,
            )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.valid, 
            batch_size=self.batch_size,
            pin_memory=self.pin_memory, 
            num_workers=self.num_workers,
            )

        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            pin_memory=self.pin_memory, 
            num_workers=self.num_workers,
            )
        return test_loader
