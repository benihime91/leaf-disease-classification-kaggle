import gc
import logging
import os

import albumentations as A
import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.core import generate_random_id
from src.layers import apply_init, replace_activs
from src.lightning.core import CassavaLightningDataModule, LightningCassava

log = logging.getLogger(__name__)


def cli_main(args: DictConfig):
    RANDOM_SEED = args.general.random_seed

    if args.general.unique_idx is None:
        args.general.unique_idx = generate_random_id()

    UNIQUE_ID = args.general.unique_idx
    MODEL_NAME = f"{args.encoder}-fold={args.curr_fold}-{UNIQUE_ID}"

    TRAIN_AUGS = A.Compose([instantiate(a) for a in args.augmentations.train])
    VALID_AUGS = A.Compose([instantiate(a) for a in args.augmentations.valid])

    # instantiate the base model architecture + activation function
    PRETRAINED_NET = instantiate(args.network.model)
    ACTIVATION_FUNC = instantiate(args.activation)

    # build the transfer learning network
    NETWORK = instantiate(
        args.network.transfer_learning_model,
        encoder=PRETRAINED_NET,
        act=ACTIVATION_FUNC,
    )

    # modify activations if activation is other than ReLU
    if args.activation._target_ != "torch.nn.ReLU":
        replace_activs(NETWORK.encoder, func=ACTIVATION_FUNC)

    # init the weights of the final untrained layer
    apply_init(NETWORK.fc, torch.nn.init.kaiming_normal_)

    # init the LightingDataModule + LightningModule
    LIGHTNING_MODEL = LightningCassava(NETWORK, conf=args)

    DATAMODULE = CassavaLightningDataModule(
        df_path=args.csv_path,
        im_dir=args.image_dir,
        curr_fold=args.curr_fold,
        train_augs=TRAIN_AUGS,
        valid_augs=VALID_AUGS,
        bs=args.batch_size,
    )

    trainer: Trainer = instantiate(args.trainer,)

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(LIGHTNING_MODEL, datamodule=DATAMODULE)

    fig = lr_finder.plot(suggest=True)

    PATH = "lr_finder_resuts.png"
    plt.imsave(PATH, fig)

    log.info(f"Suggested LR's : {lr_finder.results}")
    log.info(f"Results saved to {PATH}")

    log.info("Cleaning up .... ")

    # clean up and free memory
    try:
        del PRETRAINED_NET
        del NETWORK
        del LIGHTNING_MODEL
        del DATAMODULE
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    except:
        pass


@hydra.main(config_path="conf", config_name="example")
def cli_hydra(cfg: DictConfig):
    cli_main(cfg)


if __name__ == "__main__":
    import warnings

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    warnings.filterwarnings("ignore")

    # run train
    cli_hydra()