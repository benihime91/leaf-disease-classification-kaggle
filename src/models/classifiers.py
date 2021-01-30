# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03b_modules.classifiers.ipynb (unless otherwise specified).

__all__ = ['CLASSIFIER_REGISTERY', 'CnnHeadV0', 'CnnHeadV1', 'CnnHeadV2']

# Cell
import torch
import torch.nn.functional as F
from fastcore.all import L, delegates, ifnone
from fvcore.common import registry
from timm.models.layers import create_classifier
from torch import nn
from torch.nn import Module

from src import _logger
from .layers import *
from .utils import *

CLASSIFIER_REGISTERY = registry.Registry("Classifiers")

# Cell
@CLASSIFIER_REGISTERY.register()
def CnnHeadV0(nf, n_out, pool_type="avg", use_conv=False, **kwargs):
    "create a classifier from timm lib"
    ls = create_classifier(nf, n_out, pool_type, use_conv)
    return nn.Sequential(*ls)

# Cell
@CLASSIFIER_REGISTERY.register()
def CnnHeadV1(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True,
              first_bn=True, lin_first=False, act_layer="default", **kwargs):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    if concat_pool:
        nf *= 2
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [
        nf] + lin_ftrs + [n_out]
    bns = [first_bn] + [True]*len(lin_ftrs[1:])
    ps = L(ps)

    if len(ps) == 1:
        ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps

    actns = [ACTIVATIONS[act_layer](inplace=True)] * (len(lin_ftrs)-2) + [None]

    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)

    layers = [pool, nn.Flatten()]

    if lin_first:
        layers.append(nn.Dropout(ps.pop(0)))

    for ni, no, bn, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
        layers += LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
    if lin_first:
        layers.append(nn.Linear(lin_ftrs[-2], n_out))
    return nn.Sequential(*layers)

# Cell
@CLASSIFIER_REGISTERY.register()
class CnnHeadV2(nn.Module):
    def __init__(self, nf, n_out, dropout=0.5, act_layer="mish", **kwargs):
        super().__init__()
        self.dropout = dropout
        self.act1 = ACTIVATIONS[act_layer](inplace=True)
        self.conv = nn.Conv2d(nf, nf, 1, 1)
        self.norm = nn.BatchNorm2d(nf)
        self.pool = GeM()

        self.fc1 = nn.Linear(nf, nf // 2)
        self.act2 = ACTIVATIONS[act_layer](inplace=True)
        self.rms_norm = RMSNorm(nf // 2)
        self.fc2 = nn.Linear(nf // 2, n_out)

    def forward(self, x):
        x = self.act1(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.dropout)
        x = self.fc1(x)

        x = self.act2(x)
        x = self.rms_norm(x)
        x = F.dropout(x, p=self.dropout / 2)
        x = self.fc2(x)
        return x