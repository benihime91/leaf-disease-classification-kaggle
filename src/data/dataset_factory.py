# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01b_data.datasests_factory.ipynb (unless otherwise specified).

__all__ = ["create_transform", "DatasetMapper"]

# Cell
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .datasets import CassavaDataset, load_data
from .transforms_factory import create_transform


# Cell
class DatasetMapper:
    "A convenince class for CassavaImageClassification task"

    def __init__(self, cfg: DictConfig):
        "Note: `cfg` has to be the global hydra config"
        self.dset_cfg = cfg.data.dataset
        self.tfm_config = cfg.augmentations
        self.cfg = cfg
        self.fold = self.dset_cfg.fold

    def generate_datasets(self):
        "generates datasets and repective transformations from HYDRA config file"
        # loads the data correspoind to the current fold
        # and do some data preprocessing
        self.data = load_data(
            self.dset_cfg.csv, self.dset_cfg.image_dir, self.fold, shuffle=True
        )

        self.train_data = self.data.loc[self.data["is_valid"] == False]
        self.valid_data = self.data.loc[self.data["is_valid"] == True]

        self.train_data = self.train_data.sample(frac=1).reset_index(
            inplace=False, drop=True
        )
        self.valid_data = self.valid_data.sample(frac=1).reset_index(
            inplace=False, drop=True
        )

        # Train Test split for validation and test dataset
        self.test_data, self.valid_data = train_test_split(
            self.valid_data,
            shuffle=True,
            test_size=0.5,
            random_state=self.cfg.training.random_seed,
            stratify=self.valid_data["label"],
        )

        self.test_data = self.test_data.sample(frac=1).reset_index(
            inplace=False, drop=True
        )
        self.valid_data = self.valid_data.sample(frac=1).reset_index(
            inplace=False, drop=True
        )

        # Loads transformations from the HYDRA config file
        self.augs_initial, self.augs_final, self.augs_valid = create_transform(
            self.tfm_config, self.cfg
        )

        # Instantiate the Datasets for Training
        self.train_ds = CassavaDataset(
            self.train_data,
            fn_col="filePath",
            label_col="label",
            transform=self.augs_initial,
            backend=self.tfm_config.backend,
        )

        self.valid_ds = CassavaDataset(
            self.valid_data,
            fn_col="filePath",
            label_col="label",
            transform=self.augs_valid,
            backend=self.tfm_config.backend,
        )

        self.test_ds = CassavaDataset(
            self.test_data,
            fn_col="filePath",
            label_col="label",
            transform=self.augs_valid,
            backend=self.tfm_config.backend,
        )

    def get_train_dataset(self):
        "returns the train dataset"
        return self.train_ds

    def get_valid_dataset(self):
        "returns the validation dataset"
        return self.valid_ds

    def get_test_dataset(self):
        "return the test dataset"
        return self.test_ds

    def get_transforms(self):
        "returns the transformations to be applied after mixmethod"
        return self.augs_final
