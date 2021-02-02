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
import os
import warnings

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks import DisableValidationBar, LogInformationCallback, WandbTask
from src.core import generate_random_id, seed_everything
from src.models import Task

warnings.filterwarnings("ignore")
OmegaConf.register_resolver("eval", lambda x: eval(x))


def main(cfg: DictConfig):
    # instantiate Wandb Logger
    wandblogger = WandbLogger(project=cfg.general.project_name, log_model=True, name=cfg.training.job_name)
    # Log Hyper-parameters to Wandb
    wandblogger.log_hyperparams(cfg)

    # set random seeds so that results are reproducible
    seed_everything(cfg.training.random_seed)

    # generate a random idx for the job
    if cfg.training.unique_idx is None:
        cfg.training.unique_idx = generate_random_id()

    uq_id = cfg.training.unique_idx
    model_name = f"{cfg.training.encoder}-fold={cfg.training.fold}-{uq_id}"

    # set up cassava image classification Task
    model = Task(cfg)

    # Set up Callbacks to assist in Training
    cbs = [
        WandbTask(),
        DisableValidationBar(),
        LogInformationCallback(),
        LearningRateMonitor(cfg.scheduler.interval),
        EarlyStopping(monitor="valid/acc", patience=cfg.training.patience, mode="max"),
    ]

    checkpointCallback = ModelCheckpoint(monitor="valid/acc", save_top_k=1, mode="max",)

    # set up trainder kwargs
    kwds = dict(checkpoint_callback=checkpointCallback, callbacks=cbs, logger=wandblogger)

    trainer = instantiate(cfg.trainer, **kwds)

    trainer.fit(model)

    # Laod in the best checkpoint and save the model weights
    checkpointPath = checkpointCallback.best_model_path
    # Testing Stage
    _ = trainer.test(verbose=False, ckpt_path=checkpointPath)

    # load in the best model weights
    model = Task.load_from_checkpoint(checkpointPath)
    
    # create model save dir to save the weights of the
    # vanilla torch-model
    os.makedirs(cfg.general.save_dir, exist_ok=True)
    path = os.path.join(cfg.general.save_dir, f"{model_name}.pt")
    # save the weights of the model
    torch.save(model.model.state_dict(), f=path)
    # upload trained weights to wandb
    wandb.save(path)

    # save the original compiles config file to wandb
    conf_path = os.path.join(cfg.general.save_dir, "cfg.yml")
    OmegaConf.save(cfg, f=conf_path)
    wandb.save(conf_path)


@hydra.main(config_path="conf", config_name="config")
def cli_hydra(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    # run train
    cli_hydra()
