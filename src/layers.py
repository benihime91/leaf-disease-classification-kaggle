# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_layers.ipynb (unless otherwise specified).

__all__ = ['AdaptiveConcatPool2d', 'Mish', 'cut_model', 'num_features_model', 'create_head', 'TransferLearningModel',
           'replace_activs', 'SnapMixTransferLearningModel']

# Cell
import torch
from torch import nn
import torch.nn.functional as F

# Cell
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d` from FastAI"

    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

# Cell
class Mish(nn.Module):
    "Mish activation"
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

# Cell
def cut_model(model: nn.Module, n: int = -2):
    "cuts `model` layers upto `n`"
    ls = list(model.children())[:n]
    encoder = nn.Sequential(*ls)
    return encoder


def num_features_model(m: nn.Module, in_chs:int = 3):
    "Return the number of output features for `m`."
    m.to('cpu')
    dummy_inp = torch.zeros((32, in_chs, 120, 120))
    dummy_out = m(dummy_inp)
    return dummy_out.size()[1]

# Cell
def create_head(nf: int, n_out: int, lin_ftrs: int = 512, act: nn.Module = nn.ReLU(inplace=True)):
    "create a custom head for a classifier from FastAI"
    lin_ftrs = [nf, lin_ftrs, n_out]

    pool = AdaptiveConcatPool2d()

    layers = [pool, nn.Flatten()]

    layers += [
        nn.BatchNorm1d(lin_ftrs[0]),
        nn.Dropout(0.25),
        act,
        nn.Linear(lin_ftrs[0], lin_ftrs[1], bias=False),
        nn.BatchNorm1d(lin_ftrs[1]),
        nn.Dropout(0.5),
        act,
        nn.Linear(lin_ftrs[1], lin_ftrs[2], bias=False),
    ]
    return nn.Sequential(*layers)

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
def replace_activs(model, func, activs: list = [nn.ReLU, nn.SiLU]):
    "recursively replace all the `activs` with `func`"
    for child_name, child in model.named_children():
        for act in activs:
            if isinstance(child, act):
                setattr(model, child_name, func)
        else:
            replace_activs(child, func)

# Cell
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

        mid = len(self.encoder)//2 + 1

        self.mid_ls = cut_model(self.encoder, mid)
        mid_dim = num_features_model(self.mid_ls)

        mcls     = nn.Linear(mid_dim, self.c)
        max_pool = nn.AdaptiveMaxPool2d((1, 1))
        mid_conv = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 1, 1), nn.ReLU())

        self.mcls= nn.Sequential(mid_conv, max_pool, nn.Flatten(), mcls)

    def mid_forward(self, xb, detach=True):
        out = self.mid_ls(xb)

        if detach: out.detach()

        return self.mcls(out)


    @property
    def encoder_class_name(self):
        return self.encoder_name

    def forward(self, xb):
        fmps = self.encoder(xb)
        x = self.pool(fmps).view(fmps.size(0), -1)
        return self.fc(x)