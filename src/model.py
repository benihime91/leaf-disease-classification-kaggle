from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from omegaconf import DictConfig
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
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class LitModel(pl.LightningModule):
    """Transfer Learning with pre-trained model
    Args:
        config: Model, optimizer, scheduler hyperparameters
        weights: weights for cross-entropy loss
    """

    def __init__(self, config: DictConfig, weights: Optional[torch.Tensor] = None):
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
            self.decoder = self._creat_head()
            self.net = BasicTransferLearningModel(self.encoder, self.decoder)

        elif not self.config.model.modifiers.use_custom_base:
            last_layer = self.config.model.modifiers.last_layer
            self.encoder._modules[last_layer].out_features = self.num_classes
            self.net = self.encoder

        self.___init_modules(self.net)

        # init loss_fn
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        self.valid_loss_fn = nn.CrossEntropyLoss()

        # init metrics
        self.metric_fn = accuracy

    def ___init_modules(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight.data)

    def _num_feats_classifier(self):
        _output = torch.zeros((2, 3, 224, 224))
        _output = self.encoder(_output)
        return _output.shape[1]

    def _creat_head(self):
        nf = self._num_feats_classifier()
        n_out = self.num_classes
        lin_ftrs = self.config.model.modifiers.linear_ftrs
        lin_ftrs = [nf, lin_ftrs, n_out]
        pool = AdaptiveConcatPool2d()
        layers = [pool, nn.Flatten()]
        layers += [
            nn.BatchNorm1d(lin_ftrs[0]),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Linear(lin_ftrs[0], lin_ftrs[1]),
            nn.BatchNorm1d(lin_ftrs[1]),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, lin_ftrs[2], bias=False),
        ]
        return nn.Sequential(*layers)

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

        loss = self.valid_loss_fn(logits, y)
        acc = self.metric_fn(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        loss = self.valid_loss_fn(logits, y)
        acc = self.metric_fn(preds, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)


# class LitModel(pl.LightningModule):
#     """
#     LightningModule wrapper for Basic Transfer Learning with pre-trained classifier
#     """

#     def __init__(self, config: DictConfig, weights: Optional[torch.Tensor] = None):
#         super().__init__()
#         # Set our class attributes
#         self.config = config
#         self.num_classes = config.model.num_classes
#         self.opt_config = config.optimizer
#         self.sch_config = config.scheduler

#         # model configuration
#         model_config = config.model

#         # init classifier
#         classifier = load_obj(model_config.class_name)(**model_config.params)

#         # classifier model output dims
#         output_dims = config.model.output_dims

#         # init base model
#         if config.model.use_custom_base:
#             base_model = nn.Sequential(
#                 nn.BatchNorm1d(output_dims),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.25),
#                 nn.Linear(output_dims, model_config.fc1, bias=False),
#                 nn.BatchNorm1d(model_config.fc1),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.25),
#                 nn.Linear(model_config.fc1, model_config.fc2, bias=False),
#                 nn.BatchNorm1d(model_config.fc2),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.5),
#                 nn.Linear(model_config.fc2, self.num_classes, bias=False),
#             )

#         else:
#             base_model = nn.Sequential(
#                 nn.Dropout(0.5),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(output_dims, self.num_classes, bias=False),
#             )

#         # transfoer learning network
#         self.net = BasicTransferLearningModel(classifier, base_model)

#         # init loss_fn
#         self.loss_fn = nn.CrossEntropyLoss(weight=weights)
#         self.valid_loss_fn = nn.CrossEntropyLoss()

#         # init metrics
#         self.metric_fn = accuracy

#     def configure_optimizers(self):
#         # paramters to optimizer
#         ps = [p for p in self.net.parameters() if p.requires_grad]

#         # init optimizer
#         opt = load_obj(self.opt_config.class_name)(ps, **self.opt_config.params)

#         # init scheduler
#         sch = load_obj(self.sch_config.class_name)(opt, **self.sch_config.params)

#         # convert scheduler to lightning format
#         sch = {
#             "scheduler": sch,
#             "interval": self.sch_config.interval,
#             "frequency": self.sch_config.frequency,
#             "monitor": self.sch_config.monitor,
#         }

#         return [opt], [sch]

#     def show_trainable_layers(self):
#         # prints all the trainable layers of the model
#         for index, (name, param) in enumerate(self.net.named_parameters()):
#             if param.requires_grad:
#                 print(index, name)

#     def freeze_classifier(self):
#         # freeze the parameters of the feature extractor
#         self.net.freeze_classifier()

#     def unfreeze_classifier(self):
#         # unfreeze the parameters of the feature extractor
#         self.net.unfreeze_classifier()

#     def forward(self, x):
#         # forward pass of the model : returns logits
#         return self.net(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss_fn(logits, y)
#         self.log("train_loss", loss, prog_bar=False)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         preds = torch.argmax(logits, dim=1)

#         loss = self.valid_loss_fn(logits, y)
#         acc = self.metric_fn(preds, y)

#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_acc", acc, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         preds = torch.argmax(logits, dim=1)

#         loss = self.valid_loss_fn(logits, y)
#         acc = self.metric_fn(preds, y)

#         self.log("test_loss", loss, on_step=False, on_epoch=True)
#         self.log("test_acc", acc, on_step=False, on_epoch=True)
