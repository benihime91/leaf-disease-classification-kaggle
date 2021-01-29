import logging
import os

import albumentations as A
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.all import *

_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)


def main(cfg: DictConfig):
    _ = seed_everything(cfg.general.random_seed)

    # instantiate the base model architecture + activation function
    if cfg.network.activation is not None:
        act_func = activation_map[cfg.network.activation]
        _logger.info(f"{act_func(inplace=True)} loaded .")
    else:
        act_func = None
        _logger.info(f"Default activation function(s) loaded .")

    # with timm models pass in act_layer args to modify the activations layers
    try:
        net = instantiate(cfg.network.model, act_layer=act_func)
    except:
        # @TODO: find a way to replace activations with mdoels from pretrainedmdeols lib
        net = instantiate(cfg.network.model)

    # build the transfer learning network
    net = instantiate(
        config=cfg.network.transfer_learning_model,
        encoder=net,
        act=act_func(inplace=True),
    )

    # init the LightingDataModule + LightningModule
    if isinstance(net, VisionTransformer):
        model = LightningVisionTransformer(net, conf=cfg)
    else:
        model = LightningCassava(net, conf=cfg)

    # set up training data pipeline
    if cfg.augmentations.backend == "albumentations":
        train_augs = A.Compose([instantiate(a) for a in cfg.augmentations.train])
        valid_augs = A.Compose([instantiate(a) for a in cfg.augmentations.valid])
        _logger.info("Loaded albumentations transformations.")
    else:
        train_augs, valid_augs = None, None

    loaders = instantiate(
        config=cfg.datamodule,
        train_augs=train_augs,
        valid_augs=valid_augs,
        default_config=cfg,
    )

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

    _logger.info("Compiling Lr-Finder results ...")
    try:
        _logger.info(f"Suggested LR's : {lr_finder.suggestion():.7f}")
    except:
        pass
    _logger.info(f"Results saved to {_path}")


@hydra.main(config_path="conf", config_name="02-01-20-seresnext50_32x4d")
def cli_hydra(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

    # run train
    cli_hydra()
