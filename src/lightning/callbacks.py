# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05a_lightning.callbacks.ipynb (unless otherwise specified).

__all__ = ['logger', 'WandbImageClassificationCallback', 'LitProgressBar', 'PrintLogsCallback']

# Cell
import sys
import time
import datetime
import logging
from collections import namedtuple
from tqdm.auto import tqdm

import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log

from .core import *
from ..core import *

# Cell
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Cell
class WandbImageClassificationCallback(pl.Callback):
    """ Custom callback to add some extra functionalites to the wandb logger """

    def __init__(
        self,
        num_batches: int = 16,
        log_train_batch: bool = False,
        log_preds: bool = False,
        log_conf_mat: bool = True,
    ):

        # class names for the confusion matrix
        self.class_names = list(conf_mat_idx2lbl.values())

        # counter to log training batch images
        self.num_bs = num_batches
        self.curr_epoch = 0

        self.log_train_batch = log_train_batch
        self.log_preds = log_preds
        self.log_conf_mat = log_conf_mat

        self.val_imgs, self.val_labels = None, None

    def on_train_start(self, trainer, pl_module, *args, **kwargs):
        try:
            # log model to the wandb experiment
            wandb.watch(models=pl_module.model, criterion=pl_module.loss_func)
        except:
            logger.info("Skipping wandb.watch --->")

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if self.log_train_batch:
            if pl_module.one_batch is None:
                logger.info(
                    f"{self.config_defaults['mixmethod']} samples not available . Skipping --->"
                )
                pass

            else:
                one_batch = pl_module.one_batch[: self.num_bs]
                train_ims = one_batch.data.to("cpu")
                trainer.logger.experiment.log(
                    {"train_batch": [wandb.Image(x) for x in train_ims]}, commit=False
                )

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if self.log_preds:
            if self.val_imgs is None and self.val_labels is None:
                self.val_imgs, self.val_labels = next(iter(pl_module.val_dataloader()))
                self.val_imgs, self.val_labels = (
                    self.val_imgs[: self.num_bs],
                    self.val_labels[: self.num_bs],
                )
                self.val_imgs = self.val_imgs.to(device=pl_module.device)

            logits = pl_module(self.val_imgs)
            preds = torch.argmax(logits, 1)
            preds = preds.data.cpu()

            ims = [
                wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                for x, pred, y in zip(self.val_imgs, preds, self.val_labels)
            ]
            log_dict = {"predictions": ims}
            wandb.log(ims, commit=False)

    def on_epoch_start(self, trainer, pl_module, *args, **kwargs):
        pl_module.val_labels_list = []
        pl_module.val_preds_list = []

    def on_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if self.log_conf_mat:
            val_preds = torch.tensor(pl_module.val_preds_list).data.cpu().numpy()
            val_labels = torch.tensor(pl_module.val_labels_list).data.cpu().numpy()
            log_dict = {
                "conf_mat": wandb.plot.confusion_matrix(
                    val_preds, val_labels, self.class_names
                )
            }
            wandb.log(log_dict, commit=False)

# Cell
class LitProgressBar(pl.callbacks.ProgressBar):
    "Custom Progressbar callback for Lightning Training"

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            dynamic_ncols=True,
        )

        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            dynamic_ncols=True,
        )

        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position + 1),
            disable=True,
            dynamic_ncols=False,
        )

        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            dynamic_ncols=True,
        )

        return bar

# Cell
class PrintLogsCallback(pl.Callback):
    "Logs Training logs to console after every epoch"

    def __init__(self):
        self.TestResult = namedtuple("TestResult", ["test_loss", "test_acc"])
        self.TrainResult = namedtuple(
            "TrainResult", ["loss", "acc", "valid_loss", "valid_acc"]
        )
        self.logger = logger

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics["train/loss_epoch"]
        train_acc = metrics["train/acc_epoch"]
        valid_loss = metrics["valid/loss"]
        valid_acc = metrics["valid/acc"]
        trn_res = self.TrainResult(
            train_loss.data.cpu().numpy().item(),
            train_acc.data.cpu().numpy().item(),
            valid_loss.data.cpu().numpy().item(),
            valid_acc.data.cpu().numpy().item(),
        )

        curr_epoch = int(trainer.current_epoch)
        self.logger.info(f"[{curr_epoch}]: (100.00% done), {trn_res}")

    def on_test_epoch_end(self, trainer, pl_module, *args, **kwargs):
        metrics = trainer.callback_metrics
        test_loss = metrics["test/loss"]
        test_acc = metrics["test/acc"]
        self.logger.info(
            f"{self.TestResult(test_loss.data.cpu().numpy().item(), test_acc.data.cpu().numpy().item())}"
        )