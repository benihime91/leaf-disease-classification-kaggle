from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch import nn
from torchvision import models
import albumentations as A

from datasets import PlantDataset

# -----------------------------------------
# Base Model for Transfer Learning
# ------------------------------------------
class TransferLearningModel(nn.Module):
    """
    Transfer Learning with pre-trained classifier

    Args:
        classifier: feature extractor for the transfer learning model
        base: base model for the transfer learning model
    """
    def __init__(self, classifier: nn.Module, base: nn.Module):
        super(TransferLearningModel, self).__init__()
        # Set our init args as class attributes
        self.classifier = classifier
        self.base = base

    def freeze_classifier(self):
        for params in self.classifier.parameters():
            params.requires_grad = False
        self.classifier.eval()

    def unfreeze_classifier(self):
        for params in self.classifier.parameters():
            params.requires_grad = True
        self.classifier.train()
    
    def forward(self, x):
        """Forward pass. Returns logits."""
        # 1. Feature extraction:
        features = self.classifier(x)
        # 2. Classifier (returns logits):
        logits = self.base(features)
        return logits

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
        Lightning Module with resnext50_32x4d backbone, SGD optimizer & CosineAnnealingWarmRestarts

        Args:
            output_dims: number of output classes
            learning_rate: learning_rate for AdamW optimizer
            weight_decay: weight_decay for AdamW optimizer
            total_steps: number of total steps to train for 
            class_weights: a tensor containing the weights for nn.CrossEntropyLoss
            hidden_dims: number of nodes of the hidden layer connecting the base and output
        """

        super().__init__()
        # Set our init args as class attributes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # # init the pretrained pytorch classifier
        classifier = models.resnext50_32x4d(pretrained=True, progress=True)

        # output dims of the last layer of the classifier
        num_ftrs = classifier.fc.out_features
        # create the base for the classifier
        base_model = nn.Sequential(
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dims, self.output_dims)
        )

        # transfoer learning network
        self.net = TransferLearningModel(classifier, base_model)
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
    def configure_optimizers(self):
        opt_fn = torch.optim.SGD
        sch_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        params = [p for p in self.net.parameters() if p.requires_grad]
        
        # init optimizer and scheduler
        opt = opt_fn(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        sch = sch_fn(opt, T_0=10, T_mult=1,)
        # convert optimizer to lightning format
        sch = {"scheduler": sch, "interval": "step", "frequency":1}
        return [opt], [sch]

    def freeze_classifier(self):
        self.net.freeze_classifier()

    def unfreeze_classifier(self):
        self.net.unfreeze_classifier()

    def forward(self, x):
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
