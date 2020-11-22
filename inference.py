# Inference utilities
from torch import nn
import torch
from torchvision import models
from typing import Union
import cv2
from tqdm.auto import tqdm
import numpy as np


class InferenceDs(torch.utils.data.Dataset):
    def __init__(self, data, transformations):
        self.df = data
        self.transforms = transformations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.filePath[idx]
        unique_idx = self.df.image_id[idx]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        return img, unique_idx



class InferenceModel(nn.Module):
    def __init__(self, classifier: nn.Module, base: nn.Module):
        super(InferenceModel, self).__init__()
        self.classifier = classifier
        self.base = base

    def forward(self, x):
        features = self.classifier(x)
        logits = self.base(features)
        return logits