# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02a_networks.ipynb (unless otherwise specified).

__all__ = ['TransferLearningModel', 'SnapMixTransferLearningModel']

# Cell
import torch
from torch import nn
import torch.nn.functional as F

from .core import *
from .layers import *

# Cell
class TransferLearningModel(nn.Module):
    "Transfer Learning with `encoder`"
    def __init__(self, encoder:nn.Module, c:int, cut:int=-2, **kwargs):
        """
        Args:
            encoder: the classifer to extract features
            c: number of output classes
            cut: number of layers to cut/keep from the encoder
            **kwargs: arguments for `create_head`
        """
        super(TransferLearningModel, self).__init__()

        self.encoder_name = encoder.__class__.__name__
        # cut layers from the encoder
        self.encoder = cut_model(encoder, cut)
        # create the custom head for the model
        feats  = num_features_model(self.encoder, in_chs=3) * 2
        self.c = c
        self.fc = create_head(feats, n_out=c, **kwargs)

    @property
    def encoder_class_name(self):
        return self.encoder_name

    def forward(self, xb):
        return self.fc(self.encoder(xb))

# Cell
#TODO: add midlevel classification branch in learning.
class SnapMixTransferLearningModel(nn.Module):
    "Transfer Learning with model to be comaptible with Snapmix"
    def __init__(self, encoder:nn.Module, c:int, cut:int=-2, **kwargs):
        """
        Args:
            encoder: the classifer to extract features
            c: number of output classes
            cut: number of layers to cut/keep from the encoder
            **kwargs: arguments for `create_head`
        """
        super(SnapMixTransferLearningModel, self).__init__()

        try   : feats  = encoder.fc.in_features
        except: feats  = encoder.classifier.in_features

        self.encoder_name = encoder.__class__.__name__
        # cut layers from the encoder
        self.encoder = cut_model(encoder, cut)
        # create the custom head for the model

        self.c    = c
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(feats, self.c)

    def mid_forward(self, xb, detach=True):
        pass

    @property
    def encoder_class_name(self):
        return self.encoder_name

    def forward(self, xb):
        fmps = self.encoder(xb)
        x = self.pool(fmps).view(fmps.size(0), -1)
        return self.fc(x)