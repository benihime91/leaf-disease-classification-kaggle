import importlib
import random
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn.modules.loss import _WeightedLoss


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    torch.manual_seed(seed)


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        """
        Upon finishing training log num_samples number
        of images and their predictions to wandb
        """
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_fit_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        examples = [
            wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
            for x, pred, y in zip(val_imgs, preds, self.val_labels)
        ]
        trainer.logger.experiment.log({"examples": examples})


class PrintCallback(pl.Callback):
    def __init__(self, log):
        self.metrics = []
        self.template = "Epoch: {}, Training Loss: {:.3f}, Validation Loss: {:.3f}, Accuracy: {:.3f}"
        self.log = log

    def on_epoch_end(self, trainer, pl_module):
        metrics_dict = trainer.callback_metrics
        train_loss = metrics_dict["train_loss"].data.cpu().numpy()
        val_loss = metrics_dict["val_loss"].data.cpu().numpy()
        val_acc = metrics_dict["val_acc"].data.cpu().numpy()
        self.log.info(
            self.template.format(pl_module.current_epoch, train_loss, val_loss, val_acc)
        )


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


class SoftTargetCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean"):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()

        elif self.reduction == "mean":
            loss = loss.mean()

        return loss

