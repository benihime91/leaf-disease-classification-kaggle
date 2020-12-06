from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

# from omegaconf import DictConfig
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


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def _num_feats(model: nn.Module, dims: int = None):
    if dims is None:
        dims = 120
    _output = torch.zeros((2, 3, dims, dims))
    _output = model(_output)
    return _output.shape[1]


def _create_head(nf: int, n_out: int, lin_ftrs: int = None, act: nn.Module = nn.ReLU):
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf, lin_ftrs, n_out]

    pool = AdaptiveConcatPool2d()

    layers = [pool, nn.Flatten()]

    layers += [
        nn.BatchNorm1d(lin_ftrs[0]),
        nn.Dropout(0.25),
        act(inplace=True),
        nn.Linear(lin_ftrs[0], lin_ftrs[1], bias=False),
        nn.BatchNorm1d(lin_ftrs[1]),
        nn.Dropout(0.5),
        act(inplace=True),
        nn.Linear(lin_ftrs[1], lin_ftrs[2], bias=False),
    ]

    return nn.Sequential(*layers)


def _cut_model(model: nn.Module, upto: int = -2):
    _layers = list(model.children())[:upto]
    feature_extractor = nn.Sequential(*_layers)
    return feature_extractor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data)


class LitModel(pl.LightningModule):
    """Transfer Learning with pre-trained model
    Args:
        config: Model, optimizer, scheduler hyperparameters
        weights: weights for cross-entropy loss
    """

    def __init__(self, config, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.num_classes = config.model.num_classes
        self.opt_config = config.optimizer
        self.sch_config = config.scheduler

        # model configuration
        model_config = config.model

        # init classifier
        self.encoder = load_obj(model_config.class_name)(**model_config.params)

        # init transfer learning network
        if self.config.model.modifiers.use_custom_base:
            # cut encoder upto the feature extractors
            self.encoder = _cut_model(self.encoder)

            self.ftrs = (
                _num_feats(self.encoder, dims=self.config.training.image_dim) * 2
            )
            self.outs = self.num_classes
            self.hls = self.config.model.modifiers.linear_ftrs
            # init decoder
            self.decoder = _create_head(self.ftrs, self.outs, self.hls)
            # init model
            self.net = BasicTransferLearningModel(self.encoder, self.decoder)

        elif not self.config.model.modifiers.use_custom_base:
            last_layer = self.config.model.modifiers.last_layer
            num_ftrs = self.encoder._modules[last_layer].in_features
            self.encoder._modules[last_layer] = nn.Linear(num_ftrs, self.num_classes)
            self.net = self.encoder

        self.net.apply(weights_init)

        # init loss_fn
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.val_loss = nn.CrossEntropyLoss()

        # init metrics
        self.metric_fn = accuracy

    def show_trainable_layers(self):
        # prints all the trainable layers of the model
        for index, (name, param) in enumerate(self.net.named_parameters()):
            if param.requires_grad:
                print(index, name)

    def freeze_classifier(self):
        # freeze the parameters of the feature extractor
        try:
            self.net.freeze_classifier()
        except:
            pass

    def unfreeze_classifier(self):
        # unfreeze the parameters of the feature extractor
        try:
            self.net.unfreeze_classifier()
        except:
            for params in self.net.parameters():
                params.requires_grad = True
            self.net.train()

    def forward(self, x):
        # forward pass of the model : returns logits
        return self.net(x)

    def configure_optimizers(self):
        # paramters to optimizer
        ps = [p for p in self.net.parameters() if p.requires_grad]

        # init optimizer
        opt = load_obj(self.opt_config.class_name)(ps, **self.opt_config.params)

        # init scheduler
        sch = load_obj(self.sch_config.class_name)(opt, **self.sch_config.params)

        # convert scheduler to lightning format
        sch = {
            "scheduler": sch,
            "interval": self.sch_config.interval,
            "frequency": self.sch_config.frequency,
            "monitor": self.sch_config.monitor,
        }

        return [opt], [sch]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.val_loss(logits, y)
        acc = self.metric_fn(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.val_loss(logits, y)
        acc = self.metric_fn(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
