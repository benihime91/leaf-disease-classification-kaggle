# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04e_models.task.ipynb (unless otherwise specified).

__all__ = ['Task']

# Cell
import warnings
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.metrics import Accuracy
from torch.utils.data import DataLoader

from src import _logger
from ..data import DatasetMapper
from .builder import Net
from ..optimizers import create_optimizer
from ..schedulers import create_scheduler

warnings.filterwarnings("ignore")

# Cell
class Task(pl.LightningModule):
    "A general Task for Cassave Leaf Disease Classification"

    def __init__(self, conf: DictConfig):
        super().__init__()

        self.trn_metric = Accuracy()
        self.val_metric = Accuracy()
        self.tst_metric = Accuracy()
        self.save_hyperparameters(conf)

        # instantiate objects
        self.model = Net(self.hparams)
        self.criterion   = instantiate(self.hparams.loss)
        self.mixfunction = instantiate(self.hparams.mixmethod)

        self.lrs= None

    def setup(self, stage: str):
        "setups datasetMapper"
        mapper = DatasetMapper(self.hparams)
        mapper.generate_datasets()

        # Loads in the repective datasets
        self.train_dset = mapper.get_train_dataset()
        self.valid_dset = mapper.get_valid_dataset()
        self.test_dset  = mapper.get_test_dataset()

        # Loads in the transformations to be applied after mixmethod
        self.final_augs = mapper.get_transforms()

    def forward(self, x: Any) -> Any:
        "call the model"
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        "The Training Step: This is where the Magic Happens !!!"
        imgs, targs = batch
        self.preds, self.labels = None, None
        # store for usage later
        self.example_input_array = imgs

        if self.mixfunction is not None:
            if self.current_epoch < self.hparams.training.mix_epochs :
                imgs = self.mixfunction(imgs, targs, model=self.model)
                logits= self.forward(imgs)
                loss= self.mixfunction.lf(logits, loss_func=self.criterion)
                acc = self.trn_metric(logits, targs)
            else:
                logits = self.forward(imgs)
                loss   = self.criterion(logits, targs)
                acc    = self.trn_metric(logits, targs)

        else:
            logits = self.forward(imgs)
            loss   = self.criterion(logits, targs)
            acc    = self.trn_metric(logits, targs)

        preds  = torch.argmax(logits, 1)
        self.labels = list(targs.data.cpu().numpy())
        self.preds  = list(preds.data.cpu().numpy())

        result_dict = {"train/loss": loss, "train/acc": acc}
        self.log_dict(result_dict, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        "The Validation Step"
        imgs, targs = batch
        self.preds, self.labels = None, None

        logits = self.forward(imgs)
        loss = self.criterion(logits, targs)
        acc = self.val_metric(logits, targs)

        preds  = torch.argmax(logits, 1)
        self.labels = list(targs.data.cpu().numpy())
        self.preds  = list(preds.data.cpu().numpy())

        result_dict = {"valid/loss": loss, "valid/acc": acc}
        self.log_dict(result_dict)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        "The Test Step"
        imgs, targs = batch
        self.preds, self.labels = None, None

        logits = self.forward(imgs)
        loss = self.criterion(logits, targs)
        acc = self.tst_metric(logits, targs)

        preds  = torch.argmax(logits, 1)
        self.labels = list(targs.data.cpu().numpy())
        self.preds  = list(preds.data.cpu().numpy())

        result_dict = {"test/loss": loss, "test/acc": acc}
        self.log_dict(result_dict)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:

        lrs = (self.hparams.training.learning_rate/self.hparams.training.lr_mult,
               self.hparams.training.learning_rate)

        lr_tuple = namedtuple("LearningRates", ["base", "head"])

        self.lrs = lr_tuple(lrs[0], lrs[1])

        epochs  = self.hparams.training.num_epochs
        steps   = len(self.train_dataloader())/ self.hparams.training.accumulate_grad_batches

        total_params = self.model.get_param_list()
        params = [
            {"params": total_params[0], "lr":lrs[0]},
            {"params": total_params[1], "lr":lrs[1]},
        ]

        optim = create_optimizer(self.hparams.optimizer, params=params)
        sched = create_scheduler(self.hparams.scheduler, optim, steps, epochs)
        return [optim], [sched]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        "returns a PyTorch DataLoader for Training"
        if self.current_epoch == self.hparams.training.mix_epochs:
            if self.mixfunction is not None:
                name = self.mixfunction.__class__.__name__
                self.mixfunction.stop()

            self.train_dset.reload_transforms(self.final_augs)
            dataloader = torch.utils.data.DataLoader(self.train_dset, **self.hparams.data.dataloader)
        else:
            dataloader = torch.utils.data.DataLoader(self.train_dset, **self.hparams.data.dataloader)
        return dataloader

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        "returns a PyTorch DataLoader for Validation"
        return torch.utils.data.DataLoader(self.valid_dset, **self.hparams.data.dataloader)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        "returns a PyTorch DataLoader for Testing"
        return torch.utils.data.DataLoader(self.test_dset, **self.hparams.data.dataloader)