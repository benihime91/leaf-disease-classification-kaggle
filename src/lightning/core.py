# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00a_lightning.core.ipynb (unless otherwise specified).

__all__ = ['params', 'CassavaLightningDataModule', 'LightningCassava', 'WandbImageClassificationCallback',
           'example_conf']

# Cell
from typing import Optional, Callable, Union
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
from ..networks import *

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

        super().__init__()
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

#TODO: add midlevel classification branch in learning.
class LightningCassava(pl.LightningModule):
    """LightningModule wrapper for `TransferLearningModel`"""
    def __init__(self, model: Union[TransferLearningModel, SnapMixTransferLearningModel], conf:dict):

        super().__init__()
        self.model = model

        # save hyper-parameters
        self.save_hyperparameters(conf)
        self.metrics_to_log=['train/loss', 'train/acc', 'valid/loss', 'valid/acc', 'test/acc', 'valid/acc']

        mixmethod = object_from_dict(self.hparams["mixmethod"])
        # set mix method and loss function to be class attributes
        self.mix_fn    = mixmethod
        self.loss_func = object_from_dict(self.hparams["loss_function"])

        log.info(f'Mixmethod : {mixmethod.__class__.__name__}')
        log.info(f'Loss Function : {self.loss_func}')

        self.val_labels_list = []
        self.val_preds_list  = []
        self.one_batch_of_image = None

    def forward(self, xb):
        return self.model(xb)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mix_fn is not None:
            x = self.mix_fn(x, y, self.model)
            y_hat = self(x)
            loss  = self.mix_fn.loss(self.loss_func, y_hat)

        else:
            y_hat = self(x)
            loss  = self.loss_func(y_hat, y)

        self.one_batch_of_image = x

        train_acc = accuracy(torch.argmax(y_hat, dim=1), y)

        self.log('train/loss', loss, on_epoch=True)
        self.log('train/acc',  train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x)
        loss  = self.loss_func(y_hat, y)

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
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        base_lr = self.hparams["learning_rate"]

        param_list = [
            {'params': self.param_list[0], 'lr': base_lr/self.hparams["lr_mult"]},
            {'params': self.param_list[1], 'lr': base_lr}
        ]

        opt = object_from_dict(self.hparams["optimizer"], params=param_list)

        if self.hparams["scheduler"] is not None:
            try:
                # for OneCycleLR set the LR so we can use LrFinder
                lr_list = [self.hparams.lr/self.hparams.lr_mult, self.hparams.lr]
                kwargs = dict(optimizer=opt, max_lr=lr_list, steps_per_epoch=len(self.train_dataloader()))
                sch = object_from_dict(self.hparams["scheduler"], **kwargs)
            except:
                sch = object_from_dict(self.hparams["scheduler"], optimizer=opt)

            # convert scheduler to lightning format
            sch = {'scheduler': sch,
                   'monitor'  : self.hparams.metric_to_track,
                   'interval' : self.hparams.step_after,
                   'frequency': self.hparams.frequency}

            return [opt], [sch]

        else: return [opt]

    @property
    def param_list(self):
        param_list = [params(self.model.encoder), params(self.model.fc)]
        return param_list

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

    def __init__(self, dm: CassavaLightningDataModule, num_batches:int = 16):
        # class names for the confusion matrix
        self.class_names = list(conf_mat_idx2lbl.values())

        # counter to log training batch images
        self.dm = dm
        self.num_bs = num_batches
        self.curr_epoch = 0

    def on_train_start(self, trainer, pl_module: LightningCassava, *args, **kwargs):
        try:
            # log model to the wandb experiment
            wandb.watch(models=pl_module.model, criterion=pl_module.loss_func)
        except:
            log.info("Skipping wandb.watch --->")

        config_defaults['train_tfms'] = train_tfms
        config_defaults['valid_tfms'] = valid_tfms

        try:
            wandb.config.update(self.config_defaults)
            log.info("wandb config updated -->")
        except:
            log.info("Skipping update wandb config -->")

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if pl_module.one_batch_of_image is None:
            log.info(f"{self.config_defaults['mixmethod']} samples not available . Skipping --->")
            pass

        else:
            one_batch = pl_module.one_batch_of_image[:self.num_bs]
            train_ims = one_batch.data.to('cpu')
            trainer.logger.experiment.log({"train_batch": [wandb.Image(x) for x in train_ims]})

    def on_epoch_start(self, trainer, pl_module: LightningCassava, *args, **kwargs):
        pl_module.val_labels_list = []
        pl_module.val_preds_list  = []

    def on_epoch_end(self, trainer, pl_module: LightningCassava, *args, **kwargs):
        val_preds  = torch.tensor(pl_module.val_preds_list).data.cpu().numpy()
        val_labels = torch.tensor(pl_module.val_labels_list).data.cpu().numpy()

        # Log confusion matrix
        wandb.log({'conf_mat': wandb.plot.confusion_matrix(val_preds,val_labels,self.class_names)})

# Cell
example_conf = dict(
    mixmethod = dict(type='src.mixmethods.SnapMix', alpha=5.0, conf_prob=1.0),
    loss_function = dict(type='src.core.LabelSmoothingCrossEntropy', eps=0.1),
    learning_rate = 1e-03,
    lr_mult = 100,
    optimizer = dict(type='torch.optim.Adam', betas=(0.9, 0.99), eps=1e-06, weight_decay=0),
    scheduler = dict(type='torch.optim.lr_scheduler.CosineAnnealingWarmRestarts', T_0=10, T_mult=2),
    metric_to_track = None,
    step_after = "step",
    frequency = 1,
)