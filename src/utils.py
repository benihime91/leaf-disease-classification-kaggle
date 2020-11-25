import importlib
import random
from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
import wandb


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
        self.template = "Epoch {} - train loss: {:.3f}  val loss: {:.3f}  accuracy: {:.3f}"
        self.log = log

    def on_epoch_end(self, trainer, pl_module):
        metrics_dict = trainer.callback_metrics
        train_loss = metrics_dict["train_loss"].data.cpu().numpy()
        val_loss = metrics_dict["val_loss"].data.cpu().numpy()
        val_acc = metrics_dict["val_acc"].data.cpu().numpy()
        self.log.info(
            self.template.format(pl_module.current_epoch, train_loss, val_loss, val_acc)
        )
