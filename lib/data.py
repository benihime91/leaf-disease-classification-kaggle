from .include import *
from .utils import *


class CassaveDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        aug_tfms: A.Compose,
        path_col: str = "filePath",
        label_col: str = "label",
    ) -> None:

        self.data = dataframe
        self.augs = aug_tfms
        self.path = path_col
        self.lbls = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file = self.data[self.path][index]
        clas = self.data[self.lbls][index]

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        clas = torch.tensor(clas)

        return img, clas


class CassavaDataModule(LightningDataModule):
    def __init__(
        self,
        datafiles: Dict[str, pd.DataFrame],
        augs: Dict[str, A.Compose],
        conf: Optional[DictConfig] = None,
        dataset_func: Optional[Callable] = None,
    ):
        self.train_data = datafiles["train"]
        self.valid_data = datafiles["valid"]
        self.test_data = datafiles["test"] or datafiles["valid"]

        self.train_augs = augs["train"]
        self.valid_augs = augs["valid"]
        self.test_augs = augs["test"] or augs["valid"]

        self.config = conf
        self.dataset_func = dataset_func

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = CassaveDataset(self.train_data, self.train_augs)
            if self.dataset_func is not None:
                self.train_ds = self.dataset_func(self.train_ds)

            self.val_ds = CassaveDataset(self.valid_data, self.valid_augs)

        if stage == "test" or stage is None:
            self.test_ds = CassaveDataset(self.test_data, self.test_augs)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        ds = DataLoader(self.train_ds, **self.config, shuffle=True)
        return ds

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        ds = DataLoader(self.val_ds, **self.config, shuffle=False)
        return ds

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        ds = DataLoader(self.test_ds, **self.config)
        return ds
