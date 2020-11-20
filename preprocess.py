# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import json
import warnings

# Setup
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

# Proprocessor Class


class Preprocessor:
    """
        Usage :

        ```
            train_csv = ... # path to the train_csv file
            json_dir = ... # path to the .json label map
            image_dir = ... # path to image folder
            processor = Preprocessor(train_csv, json_dir, image_dir, 5,)
            processor._shuffle_and_create_folds(shuffle=True)

            # Grab a praticular fold
            fold_num = 0
            trainFold, valFold = processor.get_fold(fold_num)

            # grab the class imbalance weights dictionary
            # with each key correspinding to the class and values
            # correspondong to the weights
            ws = processor.weights()

        ```

        """

    def __init__(self, csv_path: str, json_path: str, image_dir: str, num_folds: int = 5):
        """
        Args : 
            1. csv_path : path to the train_csv file
            2. json_path: path to the .json label map
            3. image_dir: path to image folder
            4. num_folds: number of stratified kfolds to generate    
        """

        # read in the pandas dataframe
        dataframe = pd.read_csv(csv_path)
        # modify filepath corresponding to each image_id of the dataframe
        dataframe["filePath"] = [
            os.path.join(image_dir, dataframe.image_id[i]) for i in range(len(dataframe))
        ]
        # creat a dummy column for the kfold
        dataframe["kfold"] = -1
        # set dataframe attribute to the class
        self.dataframe = dataframe

        # read in the label map
        label_map = self._read_label_map(json_path)
        keys = list(label_map.keys())
        keys = [int(k) for k in keys]
        # set label_map attribute
        vals = list(label_map.values())
        self.label_map = {keys[i]: vals[i] for i in range(len(keys))}

        # instantiate stratified kfold
        self.num_folds = num_folds
        self.skf = StratifiedKFold(n_splits=self.num_folds, random_state=42)

        # weights for class imbalance
        self.weights = self._get_weights(self.dataframe)

    def _get_weights(self, df):
        "weights for class imbalance"
        weights = {}
        for i in range(5):
            # sample size / (num classes * class frequency)
            weights[i] = len(df)/(5 * len(df.loc[df.label == i]))
        return weights

    def _read_label_map(self, json_path):
        with open(json_path) as jfile:
            label_map = json.loads(jfile.read())
        return label_map

    def _shuffle_and_create_folds(self, shuffle=True):
        # shuffle the dataframe
        if shuffle:
            _df = self.dataframe.sample(frac=1).reset_index(drop=True)
        else:
            _df = self.dataframe

        # grab the targets
        _targets = _df.label.values
        # create cross-validation folds
        for _fold, (train_idx, val_idx) in enumerate(self.skf.split(_df, _targets)):
            _df.loc[val_idx, 'kfold'] = _fold

        # replace the original dataframe
        self.dataframe = _df

    def get_fold(self, fold_num: int):
        assert fold_num < self.num_folds, "fold_num should be less than num_folds"
        _train_fold = self.dataframe.loc[self.dataframe.kfold != fold_num]
        _val_fold = self.dataframe.loc[self.dataframe.kfold == fold_num]
        return _train_fold, _val_fold
