import random

import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .utils import onehot, rand_bbox


class PlantDataset(Dataset):
    """
    Image classification dataset.

    Args:
        dataframe: dataframe with image_id and labels
        transformations: albumentation transformations
    """

    def __init__(self, dataframe, transformations):
        self.df = dataframe
        self.transforms = transformations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.filePath[idx]
        target = self.df.label[idx]

        # Read an image with OpenCV
        img = cv2.imread(image_id)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply transformations to the image
        img = self.transforms(image=img)["image"]

        return img, target


class CutMixDatasetWrapper(Dataset):
    """
    A torch.utils.Dataset wrapper which generated cutmix samples
    """

    def __init__(self, dataset, num_class, num_mix=1, beta=1.0, prob=1.0):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2])
            )
            lb_onehot = lb_onehot * lam + lb2_onehot * (1.0 - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class LitDataModule(pl.LightningDataModule):
    """
    LightningDataModule wrapper for Image classification dataset.

    Args:
        df_train: train dataframe with image_id and labels
        df_valid: valid dataframe with image_id and labels
        df_test: test dataframe with image_id and labels
        transformations: Dictionary containing albumentation transformations for train/valid/test
        config: DictConfig containing arguments for DataLoader (batch_size, pin_memory, num_workers)
    """

    def __init__(self, df_train, df_valid, df_test, transforms, config):
        super().__init__()
        # Set our init args as class attributes
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        # albumentations transformations
        self.transforms = transforms
        # dataloader options
        self.config = config
        # self.batch_size = config.batch_size
        # self.pin_memory = config.pin_memory
        # self.num_workers = config.num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train = PlantDataset(self.df_train, self.transforms["train"])
            self.valid = PlantDataset(self.df_valid, self.transforms["valid"])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = PlantDataset(self.df_test, self.transforms["test"])

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, **self.config)

    def val_dataloader(self):
        return DataLoader(self.valid, **self.config)

    def test_dataloader(self):
        return DataLoader(self.test, **self.config)
