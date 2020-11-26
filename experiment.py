"""
This script runs the experiment with given albumentations augmentations.
For mixup and cutmix have a look at : cutmix.py & mixup.py
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

from src.data import LitDataModule
from src.model import LitModel
from src.preprocess import Preprocessor
from src.utils import PrintCallback, load_obj, set_seed


def run(config: DictConfig, logger=None):
    """Runs the training"""

    # -------------------- set up seed, wandb, logger --------------- #
    # init logger
    if logger is None:
        logger = logging.getLogger(__name__)

    # init random seed
    set_seed(config.training.seed)
    logger.info(f"using seed {config.training.seed}")

    # login to wandb
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

    # ---------------------- data preprocessing ---------------------- #

    logger.info("Prepare Training/Validation/Test Datasets.")

    # init preprocessor
    processor = Preprocessor(
        config.csv_dir, config.json_dir, config.image_dir, num_folds=5
    )

    # set the dataframe of Preprocessor to the the fold_csv
    df = pd.read_csv(config.fold_csv_dir)
    df.filePath = [
        os.path.join(config.image_dir, df.image_id[i]) for i in range(len(df))
    ]
    processor.dataframe = df

    # generate data for given fold
    fold_num = config.fold_num

    # init folds for train/valid/test data
    trainFold, valFold = processor.get_fold(fold_num)
    testFold, valFold = train_test_split(valFold, stratify=valFold.label, test_size=0.5)

    trainFold.reset_index(drop=True, inplace=True)
    testFold.reset_index(drop=True, inplace=True)
    valFold.reset_index(drop=True, inplace=True)

    # init weights for loss function
    weights = None
    if config.use_weights:
        weights = processor.weights
        weights = torch.tensor(list(weights.values()))
        weights = 1 - weights
        weights = weights.div(sum(weights))

    # ---------------------- init lightning datamodule ---------------------- #

    # init albumentation transformations
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
    dm = LitDataModule(trainFold, valFold, testFold, tfms, dl_config)
    dm.setup()

    # set training total steps
    config.training.total_steps = (
        len(dm.train_dataloader()) * config.training.num_epochs
    )

    # ---------------------- init lightning trainer ---------------------- #

    trainer_cfg = config.lightning
    # init lightning callbacks
    chkpt = pl.callbacks.ModelCheckpoint(**trainer_cfg.model_checkpoint)

    cb_config = config.lightning.callbacks
    cbs = [load_obj(module.class_name)(**module.params) for module in cb_config]
    cbs.append(PrintCallback(log=logger))

    # init trainer
    _args = trainer_cfg.init_args
    trainer = pl.Trainer(
        callbacks=cbs, checkpoint_callback=chkpt, logger=wb_logger, **_args
    )

    # ----------------- init lightning-model ------------------------------ #

    model = LitModel(config, weights=weights)

    # freeze/unfreeze the feature extractor of the model
    model.unfreeze_classifier()

    # log model topology to wandb
    wb_logger.watch(model.net)

    model_name = config.model.params.model_name or config.model.class_name

    if not config.model.use_custom_base:
        logger.info(f"init from base net {model_name} without custom classifier.")
    else:
        logger.info(f"init from base net {model_name} with custom classifier.")

    logger.info(f"Using {config.optimizer.class_name} optimizer.")
    logger.info(
        f"Learning Rate: {config.optimizer.params.lr}, Weight Decay: {config.optimizer.params.weight_decay}"
    )

    logger.info(f"Using {config.scheduler.class_name} scheduler")

    logger.info(f"Train dataset size: {len(dm.train_dataloader())} .")
    logger.info(f"OOF Validation dataset size: {len(dm.val_dataloader())} .")
    logger.info(f"OOF Test dataset size:: {len(dm.test_dataloader())} .")

    # ------------------------------ start ---------------------------------- #

    tr_config = config.training
    logger.info(
        f"Training over {tr_config.num_epochs} epochs ~ {tr_config.total_steps} steps"
    )

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, datamodule=dm)

    # Compute metrics on test dataset
    _ = trainer.test(model, datamodule=dm, ckpt_path=chkpt.best_model_path)

    PATH = chkpt.best_model_path  # path to the best performing model
    WEIGHTS_PATH = config.training.model_save_dir

    # init best model
    logger.info(f"Restoring best model weights from {PATH}")
    params = {"config": config, "weights": weights}

    loaded_model = model.load_from_checkpoint(PATH, **params)
    torchmodel = loaded_model.net

    torch.save(torchmodel.state_dict(), WEIGHTS_PATH)

    del torchmodel
    del loaded_model

    # upload the weights file to wandb
    wandb.save(WEIGHTS_PATH)

    # upload the full config file to wandb
    conf_pth = "full_config.yaml"
    OmegaConf.save(config, f=conf_pth)

    wandb.save(conf_pth)

    logger.info(f"Torch model weights saved to {WEIGHTS_PATH}")

    wandb.finish()
    # ------------------------------ end ---------------------------------- #
