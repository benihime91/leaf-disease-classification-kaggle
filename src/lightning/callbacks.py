# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05a_lightning.callbacks.ipynb (unless otherwise specified).

__all__ = ['WandbImageClassificationCallback', 'DisableValidationBar', 'PrintLogsCallback', 'DisableProgressBar',
           'ConsoleLogger']

# Cell
import os
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
from pytorch_lightning.core.memory import ModelSummary, get_human_readable_count

from ..all import *

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
            pass

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if self.log_train_batch:
            if pl_module.one_batch is None:
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
class DisableValidationBar(pl.callbacks.ProgressBar):
    "Custom Progressbar callback for Lightning Training which disables the validation bar"

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
    TrainResult = namedtuple("TrainOutput", ["loss", "acc", "valid_loss", "valid_acc"])
    TestResult = namedtuple("TestOutput", ["test_loss", "test_acc"])

    logger = logging.getLogger("_train_")

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics["train/loss_epoch"]
        train_acc = metrics["train/acc_epoch"]
        valid_loss = metrics["valid/loss"]
        valid_acc = metrics["valid/acc"]
        trn_res = self.TrainResult(
            round(train_loss.data.cpu().numpy().item(), 3),
            round(train_acc.data.cpu().numpy().item(), 3),
            round(valid_loss.data.cpu().numpy().item(), 3),
            round(valid_acc.data.cpu().numpy().item(), 3),
        )

        curr_epoch = int(trainer.current_epoch)
        self.logger.info(f"EPOCH {curr_epoch}: {trn_res}")

    def on_test_epoch_end(self, trainer, pl_module, *args, **kwargs):
        metrics = trainer.callback_metrics
        test_loss = metrics["test/loss"]
        test_acc = metrics["test/acc"]
        self.logger.info(
            f"{self.TestResult(round(test_loss.data.cpu().numpy().item(), 2), round(test_acc.data.cpu().numpy().item(), 2))}"
        )

# Cell
class DisableProgressBar(pl.callbacks.ProgressBar):
    "Custom Progressbar callback for Lightning Training which disables the validation bar"

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=True,
            dynamic_ncols=True,
        )

        return bar

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=True,
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
            disable=True,
            dynamic_ncols=True,
        )

        return bar

# Cell
class ConsoleLogger(pl.Callback):
    "Fancy logger for console-logging"
    trn_res = namedtuple("TrainOutput", ["loss", "acc", "val_loss", "val_acc"])
    tst_res = namedtuple("TestOutput", ["test_loss", "test_acc"])
    curr_step = 0
    has_init = False
    logger = logging.getLogger("_train_")

    def __init__(self, print_every: int = 50):
        self.print_every = print_every

    def on_train_start(self, trainer, pl_module, *args, **kwargs):
        if not self.has_init:
            cfg = pl_module.hparams

            self.log_line()
            self.log_msg(f"Model:")
            self.log_msg(
                f" - model_class: {str(cfg.network.transfer_learning_model._target_).split('.')[-1]}"
            )
            self.log_msg(f" - base_model: {cfg.encoder}")
            summary = ModelSummary(pl_module)
            self.log_msg(
                f" - total_parameters: {get_human_readable_count(summary.param_nums[0])}"
            )
            self.log_line()

            self.log_msg(f"Dataset:")
            self.log_msg(f" - path: {os.path.relpath(cfg.datamodule.im_dir)}")
            self.log_msg(f" - validation_fold: {str(cfg.datamodule.curr_fold)}")
            self.log_msg(
                f" - {len(pl_module.train_dataloader())} train + {len(pl_module.val_dataloader())} valid + {len(pl_module.test_dataloader())} test batches"
            )
            self.log_line()

            self.log_msg(f"Parameters:")
            self.log_msg(f" - input_dimensions: {(cfg.image_dims, cfg.image_dims, 3)}")
            self.log_msg(f" - max_epochs: {trainer.max_epochs}")
            self.log_msg(f" - mini_batch_size: {str(cfg.datamodule.bs)}")
            self.log_msg(f" - accumulate_batches: {trainer.accumulate_grad_batches}")
            self.log_msg(f" - optimizer: {str(cfg.optimizer._target_).split('.')[-1]}")
            self.log_msg(f" - learning_rates: {str(pl_module.lr_list)}")
            self.log_msg(f" - weight_decay: {str(cfg.optimizer.weight_decay)}")
            self.log_msg(
                f" - lr scheduler: {str(cfg.scheduler.function._target_).split('.')[-1]}"
            )
            self.log_msg(f" - gradient_clipping: {trainer.gradient_clip_val}")
            self.log_msg(f" - loss_function: {pl_module.loss_func}")

            has_init = True

        self.log_line()
        self.log_msg("STAGE: TRAIN / VALIDATION")
        self.log_line()
        self.log_msg(
            f"Model training base path: {os.path.relpath(trainer.checkpoint_callback.dirpath)}"
        )

        self.log_line()
        self.log_msg(f"Device: {pl_module.device}")

    def on_epoch_start(self, *args, **kwargs):
        self.log_line()

    def on_train_epoch_start(self, *args, **kwargs):
        # resets the current step
        self.curr_step = 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.curr_step % self.print_every == 0:
            ep = trainer.current_epoch
            tots = len(pl_module.train_dataloader())
            _stp_metrics = trainer.callback_metrics
            _stp_loss = _stp_metrics["train/loss_step"]
            _stp_acc = _stp_metrics["train/acc_step"]

            self.log_msg(
                f"epoch - {ep} - iteration {self.curr_step + 1}/{tots+1} - loss {_stp_loss:.3f} - acc {_stp_loss:.3f}"
            )

        self.curr_step += 1

    def on_epoch_end(self, trainer, pl_module, *args, **kwargs):
        metrics = trainer.callback_metrics

        train_loss = metrics["train/loss_epoch"]
        train_acc = metrics["train/acc_epoch"]

        valid_loss = metrics["valid/loss"]
        valid_acc = metrics["valid/acc"]

        _res = self.trn_res(
            round(train_loss.data.cpu().numpy().item(), 3),
            round(train_acc.data.cpu().numpy().item(), 3),
            round(valid_loss.data.cpu().numpy().item(), 3),
            round(valid_acc.data.cpu().numpy().item(), 3),
        )

        curr_epoch = int(trainer.current_epoch)
        self.log_line()
        self.log_msg(f"EPOCH {curr_epoch}: {_res}")

    def on_fit_end(self, *args, **kwargs):
        self.log_line()

    def on_test_start(self, trainer, pl_module, *args, **kwargs):
        self.has_init = False
        self.log_line()
        self.log_msg("STAGE: TEST")
        self.log_line()
        self.log_msg(
            f"Model testing base path: {os.path.relpath(trainer.checkpoint_callback.dirpath)}"
        )
        self.log_line()
        self.log_msg(f"Device: {pl_module.device}")
        self.log_line()

    def on_test_epoch_end(self, trainer, pl_module, *args, **kwargs):
        metrics = trainer.callback_metrics
        test_loss = metrics["test/loss"]
        test_acc = metrics["test/acc"]
        self.log_msg(
            f"{self.tst_res(round(test_loss.data.cpu().numpy().item(), 2), round(test_acc.data.cpu().numpy().item(), 2))}"
        )

    def log_line(self):
        self.logger.info("-" * 70)

    def log_msg(self, msg: str):
        self.logger.info(msg)