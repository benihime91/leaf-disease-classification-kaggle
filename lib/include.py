# numerical libs
import math
import numpy as np
import random
import PIL
from PIL import Image
import cv2
import matplotlib

# std libs
import os
import collections
from collections import OrderedDict
import random
import wandb
import uuid
import pandas as pd
from typing import *
from omegaconf import OmegaConf, DictConfig
import argparse
import hydra
from hydra.experimental import (
    initialize,
    initialize_config_dir,
    initialize_config_module,
)
import warnings
import logging
import importlib

# required imports
import torch
from torch import nn
from torch import optim
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.modules.loss import _WeightedLoss
import torchvision as tvision
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.metrics import functional as FM
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm import create_model

warnings.filterwarnings("ignore")


def set_seed(seed: Optional[int] = 42) -> int:
    "sets random seed for the experment"
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def generate_random_id() -> str:
    "generates a random id for the experminet"
    idx = uuid.uuid1()
    idx = str(idx).split("-")[0]
    return idx

