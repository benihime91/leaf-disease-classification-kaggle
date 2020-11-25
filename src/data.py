import torch
import cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


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


class LitDatatModule(pl.LightningDataModule):
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
