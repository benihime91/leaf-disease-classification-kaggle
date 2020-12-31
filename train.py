import logging
import os

import albumentations as A
import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import gc

from src.core import *
from src.layers import *
from src.lightning.callbacks import *
from src.lightning.core import *
from src.networks import *

log = logging.getLogger(__name__)

# from - https://stackoverflow.com/questions/56967754/how-to-store-and-retrieve-a-dictionary-of-tuples-in-config-parser-python
def parse_float_tuple(input) -> tuple:
    return tuple(float(k.strip()) for k in input[1:-1].split(","))


def train(cfg: DictConfig) -> None:
    "main training function"

    # break up the configs
    default_config = cfg.config
    model_hparams = cfg.hparams
    net_config = cfg.network
    aug_config = cfg.augmentations

    random_seed = seed_everything(default_config.random_seed)

    if default_config.unique_idx is None:
        idx = generate_random_id()
        default_config.unique_idx = idx

    trn_aug_list = [instantiate(augs) for augs in aug_config.train]
    valid_aug_list = [instantiate(augs) for augs in aug_config.valid]

    TRAIN_AUGS = A.Compose(trn_aug_list)
    VALID_AUGS = A.Compose(valid_aug_list)

    MODEL_NAME = f"{default_config.encoder}-fold={default_config.curr_fold}-{default_config.unique_idx}"

    # initate the model architecture
    # for snapmix we will call BasicTransferLearningModel class to init a model
    # suitable for snapmix, we can also use TransferLearningModel class to init
    # a model similar to the model created by the fast.ai cnn_learner func
    encoder = instantiate(net_config.model)
    activ = instantiate(default_config.activation)

    model = instantiate(net_config.transfer_learning_model, encoder=encoder, act=activ)

    # replace activation functions if not ReLU
    if default_config.activation._target_ != "torch.nn.ReLU":
        replace_activs(model.encoder, func=activ)

    # init the weights of the final untrained layer
    apply_init(model.fc, torch.nn.init.kaiming_normal_)

    # convert betas to tuple floats
    # if beta values are given
    try:
        model_hparams.optimizer.betas = parse_float_tuple(model_hparams.optimizer.betas)
    except:
        pass

    # init the LightingDataModule + LightningModule
    # wrap the model in LightningModule
    litModel = LightningCassava(model=model, conf=model_hparams)

    # load the lightning datamodule
    litdm = CassavaLightningDataModule(
        default_config.csv_path,
        default_config.image_dir,
        curr_fold=default_config.curr_fold,
        train_augs=TRAIN_AUGS,
        valid_augs=VALID_AUGS,
        bs=default_config.batch_size,
        num_workers=0,
    )

    # initialize pytorch_lightning Trainer + Callbacks
    callbacks = [
        LitProgressBar(),  # custom progress bar callback to stop tqdm from printing new lines
        PrintLogsCallback(),  # prints the logs after each epoch
        pl.callbacks.LearningRateMonitor(model_hparams.step_after),  # monitors the learning-rates(s)
        WandbImageClassificationCallback(litdm, default_config=default_config),  # supercharge wandb
    ]

    chkpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid/acc",
        save_top_k=1,
        mode="max",
        filename=os.path.join(cfg.save_dir, MODEL_NAME),
    )

    # wandb logger
    wb_logger = pl.loggers.WandbLogger(
        project=default_config.project_name, log_model=True
    )

    # Making a Trainer
    trainer = instantiate(
        cfg.trainer,
        checkpoint_callback=chkpt_callback,
        callbacks=callbacks,
        logger=wb_logger,
    )

    # start the training job
    trainer.fit(litModel, datamodule=litdm)

    # automatically loads in the best model weights
    # according to metric in checkpoint callback
    results = trainer.test(datamodule=litdm, ckpt_path="best", verbose=False)
    log.info(results)

    # create model save dir
    os.makedirs(cfg.save_dir, exist_ok=True)
    path = os.path.join(cfg.save_dir, f"{MODEL_NAME}.pt")

    # save the weights of the model
    litModel.save_model_weights(path)

    # upload weights to wandb
    wandb.save(path)

    # clean up and free memory
    try:
        del encoder
        del litModel
        del litdm
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    except:
        pass


@hydra.main(config_path="conf", config_name="example")
def cli_main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    import warnings

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

    warnings.filterwarnings("ignore")

    # run train
    cli_main()
