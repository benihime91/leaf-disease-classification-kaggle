# Inference utilities
from torch import nn
import torch
from torchvision import models
from typing import Union
import cv2
from tqdm.auto import tqdm


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
        return self.base(self.classifier(x))


class Predictor:
    def __init__(self, model:nn.Module, device:Union[str, torch.device]):
        self.model = model
        self.device = device

        if self.model.device != device:
            self.model.to(device)
        self.model.eval()

    def _predict_fn(self, batch):
        image, unique_idx = batch
        
        image = image.to(self.device)
        
        logits = self.model(image)
        preds  = torch.argmax(logits, -1)
        preds  = preds.cpu().detach().numpy()
        return list(preds), list(unique_idx)

    def predict(self, dataloader:torch.utils.data.DataLoader):
        batch_preds = []
        batch_idxs  = []

        for batch in tqdm(dataloader):
            preds, unique_idx = _predict_fn(batch)
            batch_preds = batch_preds + preds
            batch_idxs = batch_idxs + unique_idx

        return batch_preds, batch_idxs







