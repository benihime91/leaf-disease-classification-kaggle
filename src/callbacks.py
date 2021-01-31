# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_callbacks.ipynb (unless otherwise specified).

__all__ = ['WandbTask', 'DisableValidationBar', 'LogInformationCallback']

# Cell
import time
from collections import namedtuple

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from timm.utils import AverageMeter
from tqdm.auto import tqdm
import copy

from src import _logger
from .core import conf_mat_idx2lbl, idx2lbl
from .models import Task

# Cell
class WandbTask(Callback):
    """ Custom callback to add some extra functionalites to the wandb logger
    Does the following:
        1. Logs the model graph to wandb.
        2. Logs confusion matrix of preds/labels for each validation epoch.
        3. Logs confusion matrix of preds/labels after testing.
    """
    class_names = list(conf_mat_idx2lbl.values())

    def on_train_start(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        try   : wandb.watch(models=pl_module.model, criterion=pl_module.criterion)
        except: pass

    # def on_validation_epoch_start(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
    #     self.labels, self.predictions = [], []

    # def on_validation_batch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
    #     self.labels = self.labels + pl_module.labels
    #     self.predictions = self.predictions + pl_module.preds

    # def on_validation_epoch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
    #     preds   = torch.tensor(self.predictions).data.cpu().numpy()
    #     labels = torch.tensor(self.labels).data.cpu().numpy()

    #     matrix = wandb.plot.confusion_matrix(preds, labels, self.class_names)
    #     wandb.log(dict(valid_confusion_matrix=matrix), commit=False)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        self.labels, self.predictions = [], []

    def on_test_batch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        self.labels = self.labels + pl_module.labels
        self.predictions = self.predictions + pl_module.preds

    def on_test_epoch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        preds   = torch.tensor(self.predictions).data.cpu().numpy()
        labels = torch.tensor(self.labels).data.cpu().numpy()

        matrix = wandb.plot.confusion_matrix(preds, labels, self.class_names)
        wandb.log(dict(test_confusion_matrix=matrix), commit=False)

# Cell
class DisableValidationBar(pl.callbacks.ProgressBar):
    "Custom Progressbar callback for Lightning Training which disables the validation bar"

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(desc="Validation sanity check", dynamic_ncols=True,)
        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(desc="Training", disable=self.is_disabled, dynamic_ncols=True,)
        return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        bar = tqdm(desc="Validating", disable=True, dynamic_ncols=False,)
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(desc="Testing", disable=self.is_disabled, dynamic_ncols=True,)
        return bar

# Cell
class LogInformationCallback(pl.Callback):
    "Logs Training loss/metric to console after every epoch"
    TrainResult = namedtuple("TrainOutput", ["loss", "acc", "val_loss", "val_acc"])
    TestResult  = namedtuple("TestOutput",  ["test_loss", "test_acc"])

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        self.batch_time=  AverageMeter()
        self.end = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        self.batch_time.update(time.time() - self.end)

    def on_epoch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        metrics = copy.copy(trainer.callback_metrics)

        train_loss = metrics["train/loss_epoch"]
        train_acc  = metrics["train/acc_epoch"]
        valid_loss = metrics["valid/loss"]
        valid_acc  = metrics["valid/acc"]

        res = self.TrainResult(
            round(train_loss.data.cpu().numpy().item(), 4),
            round(train_acc.data.cpu().numpy().item(),  4),
            round(valid_loss.data.cpu().numpy().item(), 4),
            round(valid_acc.data.cpu().numpy().item(),  4),
        )

        curr_epoch = int(pl_module.current_epoch)
        total_epoch = int(trainer.max_epochs)
        _logger.info(f"Train: [ {curr_epoch}/{total_epoch}] Time: {self.batch_time.val:.3f} ({self.batch_time.avg:.3f}) {res}")

    def on_test_epoch_start(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        self.batch_time = AverageMeter()
        self.end = time.time()

    def on_test_batch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        self.batch_time.update(time.time() - self.end)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: Task, *args, **kwrags) -> None:
        metrics = trainer.callback_metrics

        test_loss = metrics["test/loss"]
        test_acc  = metrics["test/acc"]

        res = self.TestResult(
            round(test_loss.data.cpu().numpy().item(), 4),
            round(test_acc.data.cpu().numpy().item(),  4))

        _logger.info(f"Test: Time: {self.batch_time.val:.3f} ({self.batch_time.avg:.3f}) {res}")