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


class Predictor:
    def __init__(self, model:nn.Module, device:Union[str, torch.device]):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
 
    def predict(self, dataloader:torch.utils.data.DataLoader):
        batch_preds = []
        batch_idxs  = []

        for batch in tqdm(dataloader, leave=False):
            image, unique_idx = batch
            image = image.to(self.device)

            logits = self.model(image)
            
            batch_preds += [torch.softmax(logits, 1).detach().cpu().numpy()]
            batch_idxs = batch_idxs + list(unique_idx)

        batch_preds = np.concatenate(batch_preds, axis=0)
        return batch_preds, batch_idxs