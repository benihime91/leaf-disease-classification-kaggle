from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.metrics.functional.classification import accuracy
from torch import nn

from .utils import load_obj


class BasicTransferLearningModel(nn.Module):
    """
    Basic Transfer Learning with pre-trained classifier

    Args:
        classifier: feature extractor for the transfer learning model
        base: base model for the transfer learning model
    """

    def __init__(self, classifier: nn.Module, base: nn.Module):
        super(BasicTransferLearningModel, self).__init__()
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


class LitModel(pl.LightningModule):
    """
    LightningModule wrapper for Basic Transfer Learning with pre-trained classifier
    """

    def __init__(self, config: DictConfig, weights: Optional[torch.Tensor] = None):
        super().__init__()
        # Set our class attributes
        self.config = config
        self.num_classes = config.model.num_classes
        self.opt_config = config.optimizer
        self.sch_config = config.scheduler

        # model configuration
        model_config = config.model

        # init classifier
        classifier = load_obj(model_config.class_name)(**model_config.params)

        # classifier model output dims
        output_dims = config.model.output_dims

        # init base model
        base_model = nn.Sequential(
            nn.BatchNorm1d(output_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(output_dims, model_config.fc1),
            nn.BatchNorm1d(model_config.fc1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(model_config.fc1, model_config.fc2),
            nn.BatchNorm1d(model_config.fc2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(model_config.fc2, self.num_classes)
        )

        # transfoer learning network
        self.net = BasicTransferLearningModel(classifier, base_model)

        # init loss_fn
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

        # init metrics
        self.metric_fn = accuracy

    def configure_optimizers(self):
        # paramters to optimizer
        ps = [p for p in self.net.parameters() if p.requires_grad]

        # init optimizer
        opt = load_obj(self.opt_config.class_name)(
            ps, **self.opt_config.params)

        # init scheduler
        sch = load_obj(self.sch_config.class_name)(
            opt, **self.sch_config.params)

        # convert scheduler to lightning format
        sch = {
            "scheduler": sch,
            "interval": self.sch_config.interval,
            "frequency": self.sch_config.frequency,
            "monitor": self.sch_config.monitor,
        }

        return [opt], [sch]

    def freeze_classifier(self):
        # freeze the parameters of the feature extractor
        self.net.freeze_classifier()

    def unfreeze_classifier(self):
        # unfreeze the parameters of the feature extractor
        self.net.unfreeze_classifier()

    def forward(self, x):
        # forward pass of the model : returns logits
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
        acc = self.metric_fn(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)
        acc = self.metric_fn(preds, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
