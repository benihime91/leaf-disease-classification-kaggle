import uuid
import pandas as pd
from typing import *

import torch
from torch import nn

from fastai.vision.all import *


# Label map for the cassava categories
idx2lbl = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

# ========================================================
# Data / Augmentation Utilities
# ========================================================


def get_dataset(pth: str, im_dir: str, curr_fold: int, shuffle: bool = True):
    "loads the dataframe and format it"
    assert curr_fold < 5
    data = pd.read_csv(pth)
    data["filePath"] = [
        os.path.join(im_dir, data["image_id"][idx]) for idx in range(len(data))
    ]
    data["is_valid"] = [data.kfold[n] == curr_fold for n in range(len(data))]
    data["label"].replace(idx2lbl, inplace=True)

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True, inplace=False)
    else:
        data = data.reset_index(drop=True, inplace=False)

    return data


class AlbumentationsTransform(RandTransform):
    "fast.ai type transformations using albumentation transform functions"
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


# ========================================================
# General Utilities
# ========================================================
def generate_random_id() -> str:
    "returns a random id for the experiment"
    idx = uuid.uuid1()
    idx = str(idx).split("-")[0]
    return idx


# ========================================================
# Layers / Modules / Activations
# ========================================================
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"

    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


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


@delegates(create_head)
class TransferLearningModel(nn.Module):
    "Transfer Learning with pre-trained encoder."

    def __init__(self, encoder, num_classes, cut=-2, init=True, **kwargs):
        super(TransferLearningModel, self).__init__()
        self.encoder = cut_model(encoder, cut)

        ftrs = num_features_model(self.encoder) * 2
        self.fc = create_head(nf=ftrs, n_out=num_classes, **kwargs)

        if init:
            apply_init(self.decoder, nn.init.kaiming_normal_)

    def forward(self, xb):
        feats = self.encoder(xb)
        logits = self.fc(feats)
        return logits


def custom_splitter(net: TransferLearningModel):
    "custom splitter for fastai Discriminative Lrs using the TransferLearningModel"
    return [params(net.encoder), params(net.fc)]


class Mish(nn.Module):
    "mish activation"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def mod_acts(model, func, activs: list = [nn.ReLU, nn.SiLU]):
    "recursively replace all the `activs` with `func`"
    for child_name, child in model.named_children():
        for act in activs:
            if isinstance(child, act):
                setattr(model, child_name, func)
        else:
            mod_acts(child, func)


@delegates(Learner)
def timm_learner(
    dls: DataLoaders,
    encoder: nn.Module,
    cut: int,
    state: str = None,
    init: bool = True,
    pretrained: bool = True,
    act: callable = nn.ReLU(inplace=True),
    modifiers: list[callable] = None,
    **kwargs
):
    "custom fastai learner using the timm library"
    c = dls.c

    model = TransferLearningModel(
        encoder,
        num_classes=c,
        cut=cut,
        act=act,
        init=init,
    )

    if modifiers is not None:
        if isinstance(modifiers, list):
            for mods in modifiers:
                mods(model)
        else:
            modifiers(model)

    if state is not None:
        model.load_state_dict(torch.load(state))

    return Learner(dls, model, **kwargs)
