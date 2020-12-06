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


class CutmixDs(Dataset):
    def __init__(self, ds, num_class, num_mix=1, beta=1.0, prob=0.6) -> None:
        self.dataset = ds
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

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


class CassavaDataModule(LightningDataModule):
    def __init__(
        self,
        datafiles: Dict[str, pd.DataFrame],
        augs: Dict[str, A.Compose],
        conf: Optional[DictConfig] = None,
        use_cutmix: bool = False,
    ):
        self.train_data = datafiles["train"]
        self.valid_data = datafiles["valid"]
        self.test_data = datafiles["test"] or datafiles["valid"]

        self.train_augs = augs["train"]
        self.valid_augs = augs["valid"]
        self.test_augs = augs["test"] or augs["valid"]

        self.config = conf
        self.use_cutmix = use_cutmix

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = CassaveDataset(self.train_data, self.train_augs)
            if self.use_cutmix:
                self.train_ds = CutmixDs(self.train_ds, prob=0.6)

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
