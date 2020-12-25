# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00a_lightning.core.ipynb (unless otherwise specified).

__all__ = ['params', 'CassavaLightningDataModule', 'LightningCassava', 'WandbImageClassificationCallback']

# Cell
from typing import Optional, Callable
import albumentations as A
import pandas as pd
import wandb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning import _logger as log
from functools import partial

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
            self.test_ds  = ImageClassificationFromDf(self.valid_df, self.valid_augs)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.bs, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.bs, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bs, num_workers=self.workers)

# Cell
class LightningCassava(pl.LightningModule):
    """LightningModule wrapper for `TransferLearningModel`"""
    def __init__(self,
                 model: TransferLearningModel = None,
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

        # remove model from hparams as file becomes too large
        self.hparams.pop('model')

        log.info(f'Using {mixmethod}')
        log.info(f'Uses {loss_func}')

        self.metrics_to_log= ['train/loss', 'train/acc',
                              'valid/loss', 'valid/acc',
                              'test/acc', 'valid/acc']

        self.val_labels_list = []
        self.val_preds_list  = []

    def forward(self, xb):
        return self.model(xb)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mix_fn is not None:
            x = self.mix_fn(x, y, self.model)
            y_hat = self(x)
            loss = self.mix_fn.loss(self.hparams.loss_func, y_hat)

        else:
            y_hat = self(x)
            loss = self.hparams.loss_func(y_hat, y)

        self.one_batch_of_image = x

        train_acc = accuracy(torch.argmax(y_hat, dim=1), y)

        self.log('train/loss', loss, on_epoch=True)
        self.log('train/acc',  train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x)
        loss  = self.hparams.loss_func(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc   = accuracy(preds, y)

        val_labels = y.data.cpu().numpy()
        val_preds  = preds.data.cpu().numpy()

        self.val_preds_list  = self.val_preds_list + list(val_preds)
        self.val_labels_list = self.val_labels_list + list(val_labels)

        metrics = {'valid/loss': loss, 'valid/acc': acc}
        self.log_dict(metrics, prog_bar=True, logger=True,)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test/acc': metrics['valid/acc'], 'test/loss': metrics['valid/loss']}
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        ps = self.param_list
        param_list = [
            {'params': ps[0], 'lr': self.hparams.lr/self.hparams.lr_mult},
            {'params': ps[1], 'lr': self.hparams.lr}
        ]
        opt_func = self.hparams.opt_func

        try   : log.info(f'Using {opt_func}')
        except: pass

        opt = opt_func(param_list)

        if self.hparams.scheduler is not None:
            sch_func = self.hparams.scheduler

            try   : log.info(f'Using {sch_func}')
            except: pass

            try:
                # for OneCycleLR set the LR so we can use LrFinder
                lr_list = [self.hparams.lr/self.hparams.lr_mult, self.hparams.lr]
                sch = sch_func(opt, max_lr=lr_list, steps_per_epoch=len(self.train_dataloader()))
            except: sch = sch_func(opt)

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
        torch.save(state, path)
        log.info(f'weights saved to {path}')

    def load_model_weights(self, path:str):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        log.info(f'weights loaded from {path}')

# Cell
class WandbImageClassificationCallback(pl.Callback):
    """ Custom callback to add some extra functionalites to the wandb logger """

    def __init__(self, num_log_train_batchs: int = 1):
        # class names for the confusion matrix
        self.class_names = list(idx2lbl.values())

        # counter to log training batch images
        self.num_batch = 0
        self.counter   = num_log_train_batchs - 1

        self.dm = dm

    def on_train_start(self, trainer, pl_module: LightningCassava):
        # log model to the wandb experiment
        wandb.watch(models=pl_module.model, criterion=pl_module.hparams.loss_func)

        keywords = pl_module.hparams['opt_func'].keywords
        mm       = pl_module.hparams['mixmethod']

        # overrride defaults log hparams to make run more meaningfull
        config_defaults = {}
        config_defaults['num_epochs']    = trainer.max_epochs
        config_defaults['num_classes']   = len(idx2lbl)
        config_defaults['loss_func']     = pl_module.hparams['loss_func'].__class__.__name__
        config_defaults['scheduler']     = pl_module.hparams['scheduler'].func.__name__
        config_defaults['opt_func']      = pl_module.hparams['opt_func'].func.__name__
        config_defaults['lr']            = pl_module.hparams['lr']
        config_defaults['lr_mult']       = pl_module.hparams['lr_mult']
        config_defaults['learning_rate'] = pl_module.hparams['lr']
        config_defaults['weight_decay']  = keywords['weight_decay']
        config_defaults['model']         = pl_module.model.encoder_class_name

        if isinstance(mm, partial): mm   = mm.func.__name__
        else                      : mm   = mm.__class__.__name__

        config_defaults['mixmethod']     = mm

        train_tfms = list(dm.train_augs.transforms)
        valid_tfms = list(dm.valid_augs.transforms)

        config_defaults['train_tfms'] = train_tfms
        config_defaults['valid_tfms'] = valid_tfms
        config_defaults['callbacks']  = [cb.__class__.__name__ for cb in trainer.callbacks]

        pl_module.logger.log_hyperparams(config_defaults)

    def on_train_batch_end(self, trainer, pl_module: LightningCassava, *args, **kwargs):
        if self.num_batch == self.counter:
            # log the training images for the 1st batch
            train_ims = pl_module.one_batch_of_image
            train_ims = train_ims.to('cpu')
            trainer.logger.experiment.log({"train_batch": [wandb.Image(x) for x in train_ims]})
            self.num_batch += 1

    def on_epoch_start(self, trainer, pl_module: LightningCassava, *args, **kwargs):
        pl_module.val_labels_list = []
        pl_module.val_preds_list  = []

    def on_epoch_end(self, trainer, pl_module: LightningCassava, *args, **kwargs):
        val_preds  = torch.tensor(pl_module.val_preds_list).data.cpu().numpy()
        val_labels = torch.tensor(pl_module.val_labels_list).data.cpu().numpy()

        # Log confusion matrix
        wandb.log({'conf_mat': wandb.plot.confusion_matrix(val_preds,val_labels,self.class_names)})
