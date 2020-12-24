# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00a_lightning.core.ipynb (unless otherwise specified).

__all__ = ['params', 'CassavaLightningDataModule', 'LightningCassava']

# Cell
from typing import Optional, Callable
import albumentations as A
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning import _logger as log

from ..core import *
from ..layers import *
from ..mixmethods import *

# Cell
def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]

# Cell
class CassavaLightningDataModule(pl.LightningDataModule):
    "lightning-datamodule for cassave leaf disease classification"
    def __init__(self, df_path:str, im_dir:str, curr_fold: int,
                 train_augs: A.Compose, valid_augs: A.Compose, bs: int = 64,
                 num_workers: int=0):

        self.df = load_dataset(df_path, im_dir, curr_fold, True)
        self.curr_fold = curr_fold
        self.train_augs, self.valid_augs = train_augs, valid_augs
        self.bs, self.workers = bs, num_workers

    def prepare_data(self):
        log.info(f'Generating data for fold: {self.curr_fold}')
        self.train_df: pd.DataFrame = self.df.loc[self.df['is_valid'] == False]
        self.valid_df: pd.DataFrame = self.df.loc[self.df['is_valid'] == True]

        self.train_df = self.train_df.reset_index(inplace=False, drop=True)
        self.valid_df = self.valid_df.reset_index(inplace=False, drop=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = ImageClassificationFromDf(self.train_df, self.train_augs)
            self.valid_ds = ImageClassificationFromDf(self.valid_df, self.valid_augs)
        if stage == "test" or stage is None:
            self.test_ds = ImageClassificationFromDf(self.valid_df, self.valid_augs)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.bs, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.bs, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bs, num_workers=self.workers)

# Cell
class LightningCassava(pl.LightningModule):
    "LightningModule wrapper for `TransferLearningModel`"
    def __init__(self, model: TransferLearningModel = None,
                 opt_func: Callable = None,
                 lr: float = 1e-03,
                 lr_mult: int = 100,
                 step_after: Optional[str] = None,
                 frequency: int = 1,
                 metric_to_track: Optional[str] = None,
                 scheduler: Optional[Callable] = None,
                 loss_func: Callable = LabelSmoothingCrossEntropy(),
                 mixmethod: Optional[Callable] = None):

        super().__init__()
        self.model = model

        if isinstance(mixmethod, partial): self.mix_fn = mixmethod()
        else                             : self.mix_fn = mixmethod

        self.save_hyperparameters()
        log.info(f'Using {mixmethod}')
        log.info(f'Uses {loss_func}')

    def forward(self, xb):  return self.model(xb)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mix_fn is not None:
            x_mix = self.mix_fn(x, y, self.model)
            y_hat = self(x_mix)
            loss = self.mix_fn.loss(self.hparams.loss_func, y_hat)

        else:
            y_hat = self(x)
            loss = self.hparams.loss_func(y_hat, y)

        self.log("train_loss", loss, prog_bar=False)
        self.log("epoch_loss", loss, prog_bar=True, logger=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.hparams.loss_func(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y)

        metrics = {'valid_loss': loss, 'accuracy': acc}
        self.log_dict(metrics, prog_bar=True, logger=True,)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['accuracy'], 'test_loss': metrics['valid_loss']}
        self.log_dict(metrics)

    def configure_optimizers(self):
        ps = self.param_list
        param_list = [
            {'params': ps[0], 'lr': self.hparams.lr/self.hparams.lr_mult},
            {'params': ps[1], 'lr': self.hparams.lr}
        ]
        opt = self.hparams.opt_func(param_list)

        if self.hparams.scheduler is not None:
            try:
                # for OneCycleLR set the LR so we can use LrFinder
                lr_list = [self.hparams.lr/self.hparams.lr_mult, self.hparams.lr]
                sch = self.hparams.scheduler(opt, max_lr=lr_list, steps_per_epoch=len(self.train_dataloader()))
            except: sch = self.hparams.scheduler(opt)

            # convert scheduler to lightning format
            sch = {'scheduler': sch,
                   'monitor'  : self.hparams.metric_to_track,
                   'interval' : self.hparams.step_after,
                   'frequency': self.hparams.frequency}

            return [opt], [sch]

        else: return [opt]

    @property
    def param_list(self):
        return [params(self.model.encoder), params(self.model.fc)]

    def save_model_weights(self, path:str):
        state = self.model.state_dict()
        torch.save(state, state)
        log.info(f'weights saved to {path}')

    def load_model_weights(self, path:str):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        log.info(f'weights loaded from {path}')