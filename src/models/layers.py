# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04b_models.layers.ipynb (unless otherwise specified).

__all__ = ['MishJitAutoFn', 'mish', 'Mish', 'ACTIVATIONS', 'AdaptiveConcatPool2d', 'NormType', 'BatchNorm', 'LinBnDrop',
           'gem', 'GeM', 'RMSNorm']

# Cell
from enum import Enum

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from fastcore.all import L, delegates
from torch import nn

# Cell
# mish activation from : https://github.com/fastai/fastai/blob/master/fastai/layers.py#L549
@torch.jit.script
def _mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)


def mish(x):
    return MishJitAutoFn.apply(x)


class Mish(nn.Module):
    "Mish activation: Inplace is required so that it is support by timm models"
    def __init__(self, inplace=True):
        super(Mish, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)

# Cell
ACTIVATIONS = dict(default=nn.ReLU, mish=Mish, silu=nn.SiLU, sigmoid=nn.Sigmoid)

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
NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

# Cell
def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

# Cell
@delegates(nn.BatchNorm2d)
def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

# Cell
class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# Cell
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps=self.eps)'

# Cell
class RMSNorm(nn.Module):
    """An implementation of RMS Normalization.
    # https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/modules/rms_norm.html#RMSNorm
    """

    def __init__(self, dimension: int, epsilon: float = 1e-8, is_bias: bool = False):
        """
        Args:
            dimension (int): the dimension of the layer output to normalize
            epsilon (float): an epsilon to prevent dividing by zero
                in case the layer has zero variance. (default = 1e-8)
            is_bias (bool): a boolean value whether to include bias term
                while normalization
        """
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))
        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = torch.sqrt(torch.mean(x ** 2, -1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        if self.is_bias:
            return self.scale * x_norm + self.bias
        return self.scale * x_norm