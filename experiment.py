import argparse
import logging
import os

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from src.data import LitDatatModule
from src.model import LitModel
from src.preprocess import Preprocessor
from src.utils import load_obj, set_seed


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        """
        Upon finishing training log num_samples number
        of images and their predictions to wandb
        """
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_fit_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        examples = [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") for x, pred, y in zip(
            val_imgs, preds, self.val_labels)]
        trainer.logger.experiment.log({"examples": examples})


def run(config: DictConfig):
    """Runs the training"""
    # init logger
    logger = logging.getLogger("lightning")

    # init random seed
    set_seed(config.training.seed)
    logger.info(f"seed = {config.training.seed}")

    # login to wandb
    wandb.login(key=config.logger.api)

    logger.info("Prepraring the datasets ....")
    # init preprocessor
    processor = Preprocessor(config.csv_dir, config.json_dir, config.image_dir, num_folds=5)

    # set the dataframe of Preprocessor to the the fold_csv
    df = pd.read_csv(config.fold_csv_dir)
    df.filePath = [os.path.join(config.image_dir, df.image_id[i]) for i in range(len(df))]
    processor.dataframe = df

    # generate data for given fold
    fold_num = config.fold_num
    logger.info(f"Initializing data for fold : {fold_num}")

    # init folds for train/valid/test data
    trainFold, valFold = processor.get_fold(fold_num)
    testFold, valFold = train_test_split(valFold, stratify=valFold.label, test_size=0.5)

    trainFold.reset_index(drop=True, inplace=True)
    testFold.reset_index(drop=True, inplace=True)
    valFold.reset_index(drop=True, inplace=True)

    logger.info(f"Number of training examples  : {len(trainFold)}")
    logger.info(f"Number of validation examples: {len(valFold)}")
    logger.info(f"Number of testing examples   : {len(testFold)}")

    # init weights for loss function
    weights = None
    if config.use_weights:
        weights = processor.weights
        weights = torch.tensor(list(weights.values()))
        weights = 1 - weights
        weights = weights.div(sum(weights))

    # init albumentation transformations
    tfms_config = config.augmentation
    trn_augs = A.Compose([load_obj(augs.class_name)(**augs.params) for augs in tfms_config.train_augs], p=1.0)
    valid_augs = A.Compose([load_obj(augs.class_name)(**augs.params) for augs in tfms_config.valid_augs], p=1.0)
    test_augs = A.Compose([load_obj(augs.class_name)(**augs.params) for augs in tfms_config.test_augs], p=1.0)

    tfms = {"train": trn_augs, "valid": valid_augs, "test": test_augs, }

    # init datamodule
    dl_config = config.training.dataloaders
    dm = LitDatatModule(trainFold, valFold, testFold, tfms, dl_config)

    # grab samples to log predictions on
    samples = next(iter(dm.val_dataloader()))
    ims, _ = samples

    # init trainer
    logger.info("Initializing pl.Trainer ... ")

    trainer_cfg = config.lightning
    # init lightning callbacks
    chkpt = pl.callbacks.ModelCheckpoint(**trainer_cfg.model_checkpoint)

    cb_config = config.lightning.callbacks
    cbs = [load_obj(module.class_name)(module.params) for module in cb_config]
    # append magePredictionLogger callback to the callback list
    cbs.append(ImagePredictionLogger(samples))

    # init wandb logger
    wb_logger = load_obj(config.logger.class_name)(**config.logger.params)

    # init trainer
    _args = trainer_cfg.init_args
    trainer = pl.Trainer(callbacks=cbs, model_checkpoint=chkpt, logger=wb_logger, **_args)

    # log the training config to wandb
    wb_logger.log_hyperparams(config)

    logger.info("Compiling model .... ")

    # init model
    model = LitModel(config, weights=weights)
    model.example_input_array = torch.zeros_like(ims)

    # freeze/unfreeze the feature extractor of the model
    model.unfreeze_classifier()

    # log model topology to wandb
    wb_logger.watch(model.net)

    # Compute metrics on test dataset
    _ = trainer.test(model, datamodule=dm, ckpt_path=chkpt.best_model_path)

    PATH = chkpt.best_model_path  # path to the best performing model
    WEIGHTS_PATH = config.training.model_save_dir

    # init best model
    params = {"config": config, "weights": weights}
    loaded_model = model.load_from_checkpoint(PATH, **params)
    torchmodel = loaded_model.net

    # save torch model state dict
    torch.save(torchmodel.state_dict(), WEIGHTS_PATH)

    # save the weights to wandb
    # WandB – Save the model checkpoint.
    # This automatically saves a file to the cloud and associates
    # it with the current run.
    wandb.save(WEIGHTS_PATH)
    # finish run
    wandb.finish()
