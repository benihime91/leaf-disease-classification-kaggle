# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_lightning.core.ipynb (unless otherwise specified).

__all__ = ['CassavaLightningDataModule', 'fetch_scheduler', 'LightningCassava', 'LightningVisionTransformer']

# Cell
import os
from typing import Dict, List
from collections import namedtuple
import logging

import albumentations as A
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hydra.utils import instantiate, call
from omegaconf import OmegaConf, DictConfig

import pytorch_lightning as pl

from ..all import *

# Cell
import warnings

warnings.filterwarnings("ignore")

# Cell
class CassavaLightningDataModule(pl.LightningDataModule):
    "lightning-datamodule for cassave leaf disease classification"

    def __init__(
        self,
        df_path: str,
        im_dir: str,
        curr_fold: int,
        train_augs: A.Compose,
        valid_augs: A.Compose,
        bs: int = 64,
        num_workers: int = 0,
    ):

        super().__init__()
        self.df = load_dataset(df_path, im_dir, curr_fold, True)
        self.train_augs = train_augs
        self.valid_augs = valid_augs

        self.bs = bs
        self.workers = num_workers

        self.curr_fold = curr_fold
        self.im_dir = im_dir
        self.logger = logging.getLogger("datamodule")

    def prepare_data(self) -> None:
        dataInfo = namedtuple("Data", ["fold", "batch_size", "im_path"])
        self.logger.info(
            f"{dataInfo(self.curr_fold, self.bs, os.path.relpath(self.im_dir))}"
        )

        self.train_df: pd.DataFrame = self.df.loc[self.df["is_valid"] == False]
        self.valid_df: pd.DataFrame = self.df.loc[self.df["is_valid"] == True]

        self.train_df = self.train_df.reset_index(inplace=False, drop=True)
        self.valid_df = self.valid_df.reset_index(inplace=False, drop=True)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = ImageClassificationFromDf(self.train_df, self.train_augs)
            self.valid_ds = ImageClassificationFromDf(self.valid_df, self.valid_augs)
        if stage == "test" or stage is None:
            self.test_ds = ImageClassificationFromDf(self.valid_df, self.valid_augs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.bs,
            num_workers=self.workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=self.bs,
            num_workers=self.workers,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.bs,
            num_workers=self.workers,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
        )

# Cell
def fetch_scheduler(optim, conf: DictConfig, litm: pl.LightningModule) -> Dict:
    "instantiates an `scheduler` for `optim` from `conf`"
    steps = len(litm.train_dataloader()) // litm.trainer.accumulate_grad_batches

    if conf.scheduler.function._target_ == "torch.optim.lr_scheduler.OneCycleLR":
        lrs = litm.lr_list
        lr_list = [lrs.encoder_lr, lrs.fc_lr]
        kwargs = dict(optimizer=optim, max_lr=lr_list, steps_per_epoch=steps)
        sch = instantiate(conf.scheduler.function, **kwargs)

    elif conf.scheduler.function._target_ == "src.opts.FlatCos":
        kwargs = dict(optimizer=optim, steps_per_epoch=steps)
        sch = instantiate(conf.scheduler.function, **kwargs)

    elif conf.scheduler.function._target_ == "src.opts.CosineAnnealingWarmupScheduler":
        kwargs = dict(optimizer=optim, steps_per_epoch=steps)
        sch = instantiate(conf.scheduler.function, **kwargs)

    elif conf.scheduler.function._target_ == "src.opts.LinearSchedulerWithWarmup":
        kwargs = dict(optimizer=optim, steps_per_epoch=steps, warmup_steps=steps)
        sch = instantiate(conf.scheduler.function, **kwargs)
    else:
        sch = instantiate(conf.scheduler.function, optimizer=opt)

    # convert scheduler to lightning format
    sch = {
        "scheduler": sch,
        "monitor": conf.scheduler.metric_to_track,
        "interval": conf.scheduler.scheduler_interval,
        "frequency": 1,
    }

    return sch

# Cell
# TODO: add midlevel classification branch in learning.
class LightningCassava(pl.LightningModule):
    """LightningModule wrapper for `TransferLearningModel`"""

    def __init__(self, model, conf: DictConfig):
        super().__init__()
        # set up logging within the lightning-module
        self._log = logging.getLogger("LitModel")

        # set up init args
        self.model = model
        self.accuracy = pl.metrics.Accuracy()
        self.save_hyperparameters(conf)

        if isinstance(self.model, VisionTransformer):
            self._log.warning("Use class src.lightning.core.LightningVisionTransformer")

        try:
            mixmethod = instantiate(self.hparams["mixmethod"])
        except:
            mixmethod = None

        if mixmethod is not None:
            if isinstance(mixmethod, SnapMix):
                assert isinstance(self.model, SnapMixTransferLearningModel)

        self.mix_fn = mixmethod
        self.loss_func = instantiate(self.hparams["loss"])

        if self.mix_fn is not None:
            self._log.info(f"Mixmethod : {self.mix_fn}")

        self._log.info(f"Loss Function : {self.loss_func}")

        self.val_labels_list = []
        self.val_preds_list = []
        self.one_batch = None

    def forward(self, xb):
        "forward meth"
        return self.model(xb)

    def training_step(self, batch, batch_idx):
        "training step for one-batch"
        x, y = batch

        if self.mix_fn is not None:
            x = self.mix_fn(x, y, self.model)
            y_hat = self(x)
            loss = self.mix_fn.loss(self.loss_func, y_hat)

        else:
            y_hat = self(x)
            loss = self.loss_func(y_hat, y)

        self.one_batch = x

        train_acc = self.accuracy(y_hat, y)

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        "Validation step for one-batch"
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # For confusion matrix purposes
        preds = torch.argmax(y_hat, 1)
        val_labels = y.data.cpu().numpy()
        val_preds = preds.data.cpu().numpy()

        self.val_preds_list = self.val_preds_list + list(val_preds)
        self.val_labels_list = self.val_labels_list + list(val_labels)

        metrics = {"valid/loss": loss, "valid/acc": acc}

        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        "test step for one-batch"
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)
        acc = self.accuracy(y_hat, y)

        metrics = {"test/loss": loss, "test/acc": acc}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimResult = namedtuple("Optimizer", ["optimizer", "scheduler", "lrs", "wd"])

        param_list = [
            {"params": self.param_list[0], "lr": self.lr_list.encoder_lr},
            {"params": self.param_list[1], "lr": self.lr_list.fc_lr},
        ]

        opt = instantiate(self.hparams.optimizer, params=param_list)
        sch = fetch_scheduler(opt, self.hparams, self)

        opt_info = optimResult(
            opt.__class__.__name__,
            sch["scheduler"].__class__.__name__,
            self.lr_list,
            self.hparams.optimizer.weight_decay,
        )
        self._log.info(f"Optimization Parameters: \n{opt_info}")
        return [opt], [sch]

    @property
    def lr_list(self) -> namedtuple:
        "returns lrs for encoder and fc of the model"
        lrs = namedtuple("Lrs", ["encoder_lr", "fc_lr"])
        encoder_lr = self.hparams.learning_rate / self.hparams.lr_mult
        fc_lr = self.hparams.learning_rate
        return lrs(encoder_lr, fc_lr)

    @property
    def param_list(self) -> List:
        "returns the list of parameters [params of encoder, params of fc]"
        param_list = [params(self.model.encoder), params(self.model.fc)]
        return param_list

    def load_state_from_checkpoint(self, path: str):
        "loads in the weights of the LightningModule from given checkpoint"
        self._log.info(f"Attempting to load checkpoint {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self._log.info(f"Successfully loaded checkpoint {path}")

        self.load_state_dict(checkpoint["state_dict"])
        self._log.info(f"Successfully loaded weights from checkpoint {path}")

    def save_model_weights(self, path: str):
        "saves weights of self.model"
        self._log.info(f"Attempting to save weights to {path}")
        state = self.model.state_dict()
        torch.save(state, path)
        self._log.info(f"Successfully saved weights to {path}")

    def load_model_weights(self, path: str):
        "loads weights of self.model"
        self._log.info(f"Attempting to load weights from {path}")
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        self._log.info(f"Successfully loaded weights from {path}")

# Cell
# @TODO: Add Snapmix support
class LightningVisionTransformer(pl.LightningModule):
    """LightningModule wrapper for `VisionTransfer`"""

    def __init__(self, model: VisionTransformer, conf: DictConfig = None):
        super().__init__()
        self.model = model
        self._log = logging.getLogger("LitModel")
        self.accuracy = pl.metrics.Accuracy()
        self.save_hyperparameters(conf)

        try:
            mixmethod = instantiate(self.hparams["mixmethod"])
        except:
            mixmethod = None

        if mixmethod is not None:
            assert not isinstance(
                mixmethod, SnapMix
            ), "Snapmix not supported in Vision Transformer"

        self.mix_fn = mixmethod
        self.loss_func = instantiate(self.hparams["loss"])

        if self.mix_fn is not None:
            self._log.info(f"Mixmethod : {self.mix_fn}")

        self._log.info(f"Loss Function : {self.loss_func}")

        self.val_labels_list = []
        self.val_preds_list = []
        self.one_batch = None

    def forward(self, xb):
        return self.model(xb)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mix_fn is not None:
            x = self.mix_fn(x, y, self.model)
            y_hat = self(x)
            loss = self.mix_fn.loss(self.loss_func, y_hat)

        else:
            y_hat = self(x)
            loss = self.loss_func(y_hat, y)

        self.one_batch = x

        train_acc = self.accuracy(y_hat, y)

        self.log("train/loss", loss, on_epoch=True)
        self.log("train/acc", train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # For confusion matrix purposes
        preds = torch.argmax(y_hat, 1)
        val_labels = y.data.cpu().numpy()
        val_preds = preds.data.cpu().numpy()

        self.val_preds_list = self.val_preds_list + list(val_preds)
        self.val_labels_list = self.val_labels_list + list(val_labels)

        metrics = {"valid/loss": loss, "valid/acc": acc}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)
        acc = self.accuracy(y_hat, y)

        metrics = {"test/loss": loss, "test/acc": acc}
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimResult = namedtuple("Optimizer", ["optimizer", "scheduler", "lrs", "wd"])

        param_list = [
            {"params": self.param_list[0], "lr": self.lr_list.encoder_lr},
            {"params": self.param_list[1], "lr": self.lr_list.fc_lr},
        ]

        opt = instantiate(self.hparams.optimizer, params=param_list)
        sch = fetch_scheduler(opt, self.hparams, self)

        opt_info = optimResult(
            opt.__class__.__name__,
            sch["scheduler"].__class__.__name__,
            self.lr_list,
            self.hparams.optimizer.weight_decay,
        )
        self._log.info(f"Optimization Parameters: \n{opt_info}")
        return [opt], [sch]

    @property
    def lr_list(self) -> namedtuple:
        "returns lrs for encoder and fc of the model"
        lrs = namedtuple("Lrs", ["encoder_lr", "fc_lr"])
        encoder_lr = self.hparams.learning_rate / self.hparams.lr_mult
        fc_lr = self.hparams.learning_rate
        return lrs(encoder_lr, fc_lr)

    @property
    def param_list(self):
        "returns the list of parameters [(params of the model - params of head), params of head]"
        model_params = params(self.model)[:-2]
        head_params = params(self.model.model.head)
        param_list = [model_params, head_params]
        return param_list

    def load_state_from_checkpoint(self, path: str):
        "loads in the weights of the LightningModule from given checkpoint"
        self._log.info(f"Attempting to load checkpoint {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self._log.info(f"Successfully loaded checkpoint {path}")

        self.load_state_dict(checkpoint["state_dict"])
        self._log.info(f"Successfully loaded weights from checkpoint {path}")

    def save_model_weights(self, path: str):
        "saves weights of self.model"
        self._log.info(f"Attempting to save weights to {path}")
        state = self.model.state_dict()
        torch.save(state, path)
        self._log.info(f"Successfully saved weights to {path}")

    def load_model_weights(self, path: str):
        "loads weights of self.model"
        self._log.info(f"Attempting to load weights from {path}")
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        self._log.info(f"Successfully loaded weights from {path}")