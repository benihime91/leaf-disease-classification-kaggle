"""
This script runs the experiment with cutmix image augmentation
"""

import logging
import os

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data import CutMixDatasetWrapper, LitDataModule
from src.model import LitModel
from src.preprocess import Preprocessor
from src.utils import PrintCallback, load_obj, set_seed, SoftTargetCrossEntropy


def run(config: DictConfig, logger=None):
    # ----------- setup experiment ------------------- #
    # modify the datamodule
    def train_dataloader(self):
        ds = CutMixDatasetWrapper(self.train, **config.cutmix)
        return DataLoader(ds, shuffle=True, **self.config)

    LitDataModule.train_dataloader = train_dataloader

    # setup logging
    if logger is None:
        logger = logging.getLogger(__name__)

    set_seed(config.training.seed)
    logger.info(f"using seed {config.training.seed}")
    wandb.login(key=config.logger.api)

    # init wandb logger
    wb_logger = load_obj(config.logger.class_name)(**config.logger.params)

    # log the training config to wandb
    # create a new hparam dictionary with the relevant hparams and
    # log the hparams to wandb
    wb_hparam = {
        "training_fold": config.fold_num,
        "input_dims": config.training.image_dim,
        "batch_size": config.training.dataloaders.batch_size,
        "optimizer": config.optimizer.class_name,
        "scheduler": config.scheduler.class_name,
        "learning_rate": config.optimizer.params.lr,
        "weight_decay": config.optimizer.params.weight_decay,
        "num_epochs": config.training.num_epochs,
        "use_loss_fn_weights": config.use_weights,
        "use_custom_base": config.model.use_custom_base,
    }
    wb_logger.log_hyperparams(wb_hparam)

    # ----------- prepare datasets ------------------- #

    logger.info("Prepare Training/Validation/Test Datasets.")

    processor = Preprocessor(config.csv_dir, config.json_dir, config.image_dir, 5)
    df = pd.read_csv(config.fold_csv_dir)
    imsdir = config.image_dir
    df.filePath = [os.path.join(imsdir, df.image_id[i]) for i in range(len(df))]

    processor.dataframe = df
    fold_num = config.fold_num
    trainFold, valFold = processor.get_fold(fold_num)
    # testFold, valFold = train_test_split(valFold, stratify=valFold.label, test_size=0.5)
    
    trainFold.reset_index(drop=True, inplace=True)
    # testFold.reset_index(drop=True, inplace=True)
    valFold.reset_index(drop=True, inplace=True)

    # init weights for loss function
    weights = None
    if config.use_weights:
        weights = processor.weights
        weights = torch.tensor(list(weights.values()))
        weights = 1 - weights
        weights = weights.div(sum(weights))

    tfms_config = config.augmentation

    trn_augs = A.Compose(
        [load_obj(augs.class_name)(**augs.params) for augs in tfms_config.train_augs],
        p=1.0,
    )
    valid_augs = A.Compose(
        [load_obj(augs.class_name)(**augs.params) for augs in tfms_config.valid_augs],
        p=1.0,
    )
    test_augs = A.Compose(
        [load_obj(augs.class_name)(**augs.params) for augs in tfms_config.test_augs],
        p=1.0,
    )

    tfms = {
        "train": trn_augs,
        "valid": valid_augs,
        "test": test_augs,
    }

    # init datamodule
    dl_config = config.training.dataloaders
    dm = LitDataModule(trainFold, valFold, valFold, tfms, dl_config)
    dm.setup()

    # set training total steps
    config.training.total_steps = (len(dm.train_dataloader()) * config.training.num_epochs)

    logger.info(f"Train dataset size: {len(dm.train_dataloader())}")
    logger.info(f"Validation dataset size: {len(dm.val_dataloader())}")

    # ----------- load lightning trainer ------------------- #

    trainer_cfg = config.lightning
    # init lightning callbacks
    chkpt = pl.callbacks.ModelCheckpoint(**trainer_cfg.model_checkpoint)

    cb_config = config.lightning.callbacks
    cbs = [load_obj(module.class_name)(**module.params) for module in cb_config]
    cbs.append(PrintCallback(log=logger))

    # init trainer
    _args = trainer_cfg.init_args
    trainer = pl.Trainer(callbacks=cbs, checkpoint_callback=chkpt, logger=wb_logger, **_args)

    # ----------- init lightning module ------------------- #
    logger.info("Build network.")
    
    model = LitModel(config, weights=weights)
    # update model loss function to soft cross entropy loss
    model.loss_fn = SoftTargetCrossEntropy(weight=weights)
    model.unfreeze_classifier()

    wb_logger.watch(model.net)

    model_name = config.model.params.model_name or config.model.class_name

    if not config.model.use_custom_base:
        logger.info(f"Init from base net: {model_name}, without custom classifier.")
    else:
        logger.info(f"Init from base net: {model_name}, with custom classifier.")

    logger.info(f"Uses {str(config.optimizer.class_name).split('.')[-1]} optimizer.")
    logger.info(f"Learning Rate: {config.optimizer.params.lr}, Weight Decay: {config.optimizer.params.weight_decay}")
    logger.info(f"Uses {str(config.scheduler.class_name).split('.')[-1]} scheduler.")
    tr_config = config.training
    logger.info(f"Training over {tr_config.num_epochs} epochs ~ {tr_config.total_steps} steps.")

    # ----------- start train/validaiton/test ------------------- #

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, datamodule=dm)

    # Compute metrics on test dataset
    _ = trainer.test(model, datamodule=dm, ckpt_path=chkpt.best_model_path)

    # ----------- finish experiment/cleanup/save weights ------------------- #

    PATH = chkpt.best_model_path  # path to the best performing model
    WEIGHTS_PATH = config.training.model_save_dir

    # init best model
    logger.info(f"Restored best model weights from {PATH}.")
    params = {"config": config, "weights": weights}

    loaded_model = model.load_from_checkpoint(PATH, **params)
    torchmodel = loaded_model.net

    torch.save(torchmodel.state_dict(), WEIGHTS_PATH)

    del torchmodel
    del loaded_model

    # upload the weights file to wandb
    wandb.save(WEIGHTS_PATH)

    # upload the full config file to wandb
    conf_pth = f"{config.run_name}.yaml"
    OmegaConf.save(config, f=conf_pth)
    logger.info(f"Saved config file {conf_pth}.")

    wandb.save(conf_pth)

    logger.info(f"Saved model {WEIGHTS_PATH}.")

    wandb.finish()
