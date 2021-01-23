import logging
import os

import albumentations as A
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.all import *


def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    seed = seed_everything(cfg.general.random_seed)

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

    # initialize pytorch_lightning
    trainer: Trainer = instantiate(cfg.trainer)

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, datamodule=loaders)

    fig = lr_finder.plot(suggest=True)

    # create directory if does not exists
    os.makedirs(cfg.general.save_dir, exist_ok=True)

    # save lr-finder plot to memory
    _path = os.path.join(cfg.general.save_dir, f"lr-finder-plot.png")
    fig.savefig(_path)

    logger.info(f"Suggested LR's : {lr_finder.suggestion().:6f}")
    logger.info(f"Results saved to {_path}")


@hydra.main(config_path="conf", config_name="02-01-20-seresnext50_32x4d")
def cli_hydra(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    # run train
    cli_hydra()
