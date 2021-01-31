# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04d_models.builder.ipynb (unless otherwise specified).

__all__ = ['build_head', 'Net']

# Cell
import timm
import torch
from fastcore.all import ifnone
from omegaconf import DictConfig
from timm.models.layers import create_classifier
from torch import nn

from src import _logger
from ..core import *
from .classifiers import *
from .layers import *
from .utils import apply_init, cut_model, num_features_model

# Cell
def build_head(cfg: DictConfig, nf, verbose=False):
    "builds a classifier for model with output `nf`"
    head = CLASSIFIER_REGISTERY.get(cfg.name)(nf=nf, **cfg.params)
    return head

# Cell
class Net(nn.Module):
    "Creates a model using the Global Config"

    def __init__(self, cfg: DictConfig, verbose=True):
        super(Net, self).__init__()
        self.base_conf = cfg.model.base_model
        self.head_conf = cfg.model.head

        # build the encoder
        if verbose:
            _logger.info("Configuration for the current model :")
            _logger.info(f" feature_extractor: {self.base_conf.name}")

            if self.base_conf.activation is not None:
                _logger.info(f" activation: {self.base_conf.activation}")

            _logger.info(f" params: {self.base_conf.params}")

            for k, v in self.head_conf.items():
                if k == "name":
                    _logger.info(f" head: {str(v)}")
                else :
                    _logger.info(f" {k}: {v}")

        if self.base_conf.activation is not None:  act = ACTIVATIONS[self.base_conf.activation]
        else                                    :  act = None

        self.encoder = timm.create_model(self.base_conf.name, act_layer=act, **self.base_conf.params)
        self.encoder = cut_model(self.encoder, -2)
        nf = num_features_model(self.encoder)
        self.head = build_head(self.head_conf, nf, verbose)


    def init_classifier(self):
        if self.clf_conf.act_layer == "default":
            apply_init(self.classifier, torch.nn.init.kaiming_normal_)
        else:
            apply_init(self.classifier, torch.nn.init.kaiming_uniform_)

    def get_head(self):
        return self.head

    def get_classifier(self):
        try:
            return self.head[-1]
        except:
            return self.head.fc2

    def forward_features(self, x: torch.Tensor):
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        return self.head(self.forward_features(x))

    def get_param_list(self):
        return [params(self.encoder), params(self.head)]