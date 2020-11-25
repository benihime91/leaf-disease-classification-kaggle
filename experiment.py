import argparse
import logging
import os
import shutil

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


def run(config: DictConfig, logger=None, print_layers:bool = False):
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

    # ---------------------- data preprocessing ---------------------- #

    # init preprocessor
    processor = Preprocessor(config.csv_dir, config.json_dir, config.image_dir, num_folds=5)

    # set the dataframe of Preprocessor to the the fold_csv
    df = pd.read_csv(config.fold_csv_dir)
    df.filePath = [os.path.join(config.image_dir, df.image_id[i]) for i in range(len(df))]
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
    trn_augs = A.Compose([load_obj(augs.class_name)(**augs.params) for augs in tfms_config.train_augs], p=1.0)
    valid_augs = A.Compose([load_obj(augs.class_name)(**augs.params) for augs in tfms_config.valid_augs], p=1.0)
    test_augs = A.Compose([load_obj(augs.class_name)(**augs.params) for augs in tfms_config.test_augs], p=1.0)

    tfms = {"train": trn_augs, "valid": valid_augs, "test": test_augs, }

    # init datamodule
    dl_config = config.training.dataloaders
    dm = LitDatatModule(trainFold, valFold, testFold, tfms, dl_config)
    dm.setup()

    logger.info(f"init dataloaders with {config.training.image_dim} image dim and batch_size of {config.training.dataloaders.batch_size}")

    # grab samples to log predictions on
    samples = next(iter(dm.val_dataloader()))
    ims, _ = samples

    # set training total steps
    config.training.total_steps = len(dm.train_dataloader()) * config.training.num_epochs

    # ---------------------- init lightning trainer ---------------------- #
    
    trainer_cfg = config.lightning
    # init lightning callbacks
    chkpt = pl.callbacks.ModelCheckpoint(**trainer_cfg.model_checkpoint)

    cb_config = config.lightning.callbacks
    cbs = [load_obj(module.class_name)(**module.params) for module in cb_config]
    # append magePredictionLogger callback to the callback list
    cbs.append(ImagePredictionLogger(samples))

    del samples

    # init wandb logger
    wb_logger = load_obj(config.logger.class_name)(**config.logger.params)

    # init trainer
    _args = trainer_cfg.init_args
    trainer = pl.Trainer(callbacks=cbs, checkpoint_callback=chkpt, logger=wb_logger, **_args)

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

    # ----------------- init lightning-model ------------------------------ #

    model = LitModel(config, weights=weights)
    model.example_input_array = torch.zeros_like(ims)
    
    if print_layers:
        model.show_trainable_layers()

    # freeze/unfreeze the feature extractor of the model
    model.unfreeze_classifier()

    # log model topology to wandb
    wb_logger.watch(model.net)

    model_name = config.model.params.model_name or config.model.class_name
    
    if not config.model.use_custom_base:
        logger.info(f"init model from {model_name} without custom base")
    else:
        logger.info(f"init model from {model_name} with custom base")

    logger.info(f"init {config.optimizer.class_name} optimizer")
    logger.info(f"init {config.scheduler.class_name} scheduler")

    # ------------------------------ start ---------------------------------- #

    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, datamodule=dm)
    
    # Compute metrics on test dataset
    _ = trainer.test(model, datamodule=dm, ckpt_path=chkpt.best_model_path)

    PATH = chkpt.best_model_path  # path to the best performing model
    WEIGHTS_PATH = config.training.model_save_dir

    # init best model
    params       = {"config": config, "weights": weights}
    loaded_model = model.load_from_checkpoint(PATH, **params)
    torchmodel   = loaded_model.net
    
    torch.save(torchmodel.state_dict(), WEIGHTS_PATH)

    
    del torchmodel
    del loaded_model
    
    # upload the weights file to wandb
    wandb.save(WEIGHTS_PATH)
    # upload the full config file to wandb
    conf_pth = "full_config.yaml"
    OmegaConf.save(conf=config, f=conf_pth)
    wandb.save(conf_pth)

    shutil.rmtree(conf_pth)

    if not config.save_pytorch_model:
        shutil.rmtree(WEIGHTS_PATH)

    if config.save_pytorch_model:
        logger.info(f"Torch model weights saved to {WEIGHTS_PATH}")
    
    wandb.finish()
    # ------------------------------ end ---------------------------------- #
