# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01a_data.datasets.ipynb (unless otherwise specified).

__all__ = ['load_data', 'pil_loader', 'cv2_loader', 'CassavaDataset']

# Cell
import os

import cv2
import pandas as pd
import torch
from fastcore.all import delegates, ifnone, store_attr
from PIL import Image
from torch.utils.data import Dataset

from src import _logger
from ..core import idx2lbl

# Cell
def load_data(pth: str, im_dir: str, curr_fold: int = 0, shuffle: bool = True) -> pd.DataFrame:
    "loads the dataframe and formats it"
    assert curr_fold < 5

    data = pd.read_csv(pth)

    data["filePath"] = [os.path.join(im_dir, data["image_id"][idx]) for idx in range(len(data))]
    data["is_valid"] = [data.kfold[n] == curr_fold for n in range(len(data))]

    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True, inplace=False)
    else:
        data = data.reset_index(drop=True, inplace=False)

    return data

# Cell
def pil_loader(path):
    "loads an image using PIL, mainly used for torchvision transformations"
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def cv2_loader(path):
    "loads an image using cv2, mainly used for albumentations transformations"
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Cell
class CassavaDataset(Dataset):
    "Create `CassavaDataset` rom `df` using `fn_col`"

    def __init__(self, df, fn_col, label_col=None, transform=None, train: bool = True, backend="torchvision"):
        store_attr("df, fn_col, label_col, transform, train, backend")
        self.df = df.copy()
        self._setup_loader()

    def _setup_loader(self):
        if self.backend == "torchvision":
            self.loader = pil_loader
        elif self.backend == "albumentations":
            self.loader = cv2_loader

    def reload_transforms(self, transform, backend=None):
        "change the transformations used after `__init__`"
        self.backend = ifnone(backend, self.backend)
        self._setup_loader()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df[self.fn_col][index]
        # load the image
        image = self.loader(image_id)

        # apply transformations
        if self.backend == "torchvision":
            image = self.transform(image)
        elif self.backend == "albumentations":
            image = self.transform(image=image)

        # returns image-tensors and Optionally target-tensors
        if self.train:
            target = self.df[self.label_col][index]
            target = torch.tensor(target)
            return image, target
        else:
            return image

    @classmethod
    @delegates(__init__)
    def from_albu_tfms(cls, df, fn_col, backend="albumentations", **kwargs):
        "Create `Dataset` from `df` using `albumentations` transformations and `cv2`"
        return cls(df, fn_col, backend=backend, **kwargs)

    @classmethod
    @delegates(__init__)
    def from_torchvision_tfms(cls, df, fn_col, backend="torchvision", **kwargs):
        "Create `Dataset` from `df` using `torchvision` transformations and `PIL`"
        return cls(df, fn_col, backend=backend, **kwargs)