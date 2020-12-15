from fastai.vision.all import *
import torch
from torch import nn
import uuid
import albumentations as A
import numpy as np

__all__ = [
    "AlbumentationsTransform",
    "generate_random_id",
    "Mish",
    "AdaptiveConcatPool2d",
    "replace_with_mish",
    "cut_model",
    "create_head",
    "custom_splitter",
    "TransferLearningModel",
]


class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    split_idx, order = None, 2

    def __init__(self, train_aug, valid_aug):
        store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))["image"]
        else:
            aug_img = self.valid_aug(image=np.array(img))["image"]
        return PILImage.create(aug_img)


def generate_random_id() -> str:
    "returns a random id for the experiment"
    idx = uuid.uuid1()
    idx = str(idx).split("-")[0]
    return idx


class Mish(nn.Module):
    "Mish activation function"

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def replace_with_mish(model: nn.Module):
    "recursively replace all ReLU/SiLU activations to Mish activation"
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU) or isinstance(child, nn.SiLU):
            setattr(model, child_name, Mish())
        else:
            replace_with_mish(child)


def cut_model(model: nn.Module, upto: int = -2) -> nn.Module:
    ls = list(model.children())[:upto]
    encoder = nn.Sequential(*ls)
    return encoder


def create_head(
    nf: int, n_out: int, lin_ftrs: int = 512, act: nn.Module = nn.ReLU(inplace=True)
):
    "create a custom head for a classifier"
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


def custom_splitter(net):
    return [params(net.encoder), params(net.decoder)]


@delegates(create_head)
class TransferLearningModel(nn.Module):
    "Transfer Learning with pre-trained encoder."

    def __init__(self, encoder, num_classes, cut=-2, **kwargs):
        super(TransferLearningModel, self).__init__()
        self.encoder = cut_model(encoder, cut)

        ftrs = num_features_model(self.encoder) * 2
        self.decoder = create_head(nf=ftrs, n_out=num_classes, **kwargs)

    def forward(self, xb):
        feats = self.encoder(xb)
        logits = self.decoder(feats)
        return logits
