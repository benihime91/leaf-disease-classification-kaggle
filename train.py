""" Training Script
This script can be used to launch a training job. The config for the training can be modified
in the conf/ directory. This script uses hydra to configure the training job. 
See (https://hydra.cc/docs/intro/)

Config can also be modified from the command line.
See (https://hydra.cc/docs/advanced/overriding_packages).

For training this script utilizes the pytorch-lightning training. Modify the default configuration of the 
trainer in conf/trainer.

Image augmentations are applied from albumentations (https://albumentations.ai/docs/). 
Modify the augmentations in conf/augmentations.

Optimizer and Scheduler config can be modified in conf/optimizer & conf/scheduler respectively.

The main aim of the scipt is to iterate over different experimentations with minimal changes.

Note: To use the lr_finder algorithm to get a good starting learning rate, run the script finder.py.
See (https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html)
"""
import logging
import os

import albumentations as A
import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from src.all import *

log = logging.getLogger(__name__)


def main(cfg: DictConfig):
    # init wandb experiment
    wb_logger = WandbLogger(project=cfg.general.project_name, log_model=True)
    wb_logger.log_hyperparams(cfg)

    seed = seed_everything(cfg.general.random_seed)

    if cfg.general.unique_idx is None:
        cfg.general.unique_idx = generate_random_id()

    uq_id = cfg.general.unique_idx
    model_name = f"{cfg.encoder}-fold={cfg.datamodule.curr_fold}-{uq_id}"

    trn_augs = A.Compose([instantiate(a) for a in cfg.augmentations.train])
    val_augs = A.Compose([instantiate(a) for a in cfg.augmentations.valid])

    # instantiate the base model architecture + activation function
    net = instantiate(cfg.network.model)
    act_func = instantiate(cfg.activation)

    # build the transfer learning network
    net = instantiate(cfg.network.transfer_learning_model, encoder=net, act=act_func,)

    # modify activations if activation is other than ReLU
    if cfg.activation._target_ != "torch.nn.ReLU":
        replace_activs(net.encoder, func=act_func)

    # init the weights of the final untrained layer
    try:
        apply_init(net.fc, torch.nn.init.kaiming_normal_)
    except:
        # for vision transformer
        apply_init(net.model.head, torch.nn.init.kaiming_normal_)

    # init the LightingDataModule + LightningModule
    if isinstance(net, VisionTransformer):
        model = LightningVisionTransformer(net, conf=cfg)
    else:
        model = LightningCassava(net, conf=cfg)

    loaders = instantiate(cfg.datamodule, train_augs=trn_augs, valid_augs=val_augs,)

    # initialize pytorch_lightning Trainer + Callbacks
    cbs = [
        WandbImageClassificationCallback(log_conf_mat=True),
        LitProgressBar(),
        PrintLogsCallback(),
        LearningRateMonitor(cfg.scheduler.scheduler_interval),
        EarlyStopping(monitor="valid/acc", mode="max"),
    ]

    chkpt_cb = ModelCheckpoint(monitor="valid/acc", save_top_k=1, mode="max",)

    _trn_kwargs = dict(checkpoint_callback=chkpt_cb, callbacks=cbs, logger=wb_logger)
    trainer: Trainer = instantiate(cfg.trainer, **_trn_kwargs)

    # Start Train + Validation
    trainer.fit(model, datamodule=loaders)

    # Laod in the best checkpoint and save the model weights
    ckpt_path = chkpt_cb.best_model_path

    # Testing Stage
    _ = trainer.test(datamodule=loaders, verbose=False, ckpt_path=ckpt_path)

    # load in the best model weights
    model.load_state_from_checkpoint(ckpt_path)

    # create model save dir
    os.makedirs(cfg.general.save_dir, exist_ok=True)
    _path = os.path.join(cfg.general.save_dir, f"{model_name}.pt")

    # save the weights of the model
    model.save_model_weights(_path)

    # upload trained weights to wandb
    wandb.save(_path)

    # save the original compiles config file to wandb
    conf_path = os.path.join(cfg.general.save_dir, "cfg.yaml")
    OmegaConf.save(cfg, f=conf_path)
    wandb.save(conf_path)


@hydra.main(config_path="conf", config_name="02-01-20-seresnext50_32x4d")
def cli_hydra(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    # run train
    cli_hydra()
