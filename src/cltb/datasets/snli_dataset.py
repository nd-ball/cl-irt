from datasets.cl_dataset import CLDataset 
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder


FNAMES = {
    "train": "snli_1.0_train.txt",
    "dev": "snli_1.0_dev.txt",
    "test": "snli_1.0_test.txt"
}


class SNLIDataset(CLDataset):
    def __init__(self,config):
        self.split = config["data_split"]
        self.data_path = config["data_path"]
        self.data_path = f"{self.data_path}/{FNAMES[self.split]}"

        self.raw_data = pd.read_csv(self.data_path, sep='\t',
                            usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        self.ids = self.raw_data.pairID
        self.examples = self.raw_data.sentence1
        self.examples2 = self.raw_data.sentence2
        
        le = LabelEncoder()
        self.labels = le.fit_transform(self.raw_data.gold_label)
        
        if config["difficulties_file"]:
            self.difficulties_file = config["difficulties_file"]
            self.difficulties = pd.read_csv(self.difficulties_file, sep=',',
                                header=None, names=['pairID', 'difficulty'])
            self.difficulties = self.difficulties.set_index('pairID')
            self.difficulties = self.difficulties.to_dict('index')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        examples = self.examples.iloc[idx, 1:]
        examples = np.array([examples])
        examples2 = self.examples2.iloc[idx, 1:]
        examples2 = np.array([examples2])
        ids = self.ids.iloc[idx, 1:]
        ids = np.array([ids])
        #difficulties = self.difficulties.iloc[idx, 1:]
        difficulties = [self.difficulties[i] for i in ids]
        difficulties = np.array([difficulties])
        labels = self.labels.iloc[idx, 1:]
        labels = np.array([labels])
        sample = {
            "ids": ids,
            "examples": examples,
            "examples2": examples2,  
            "difficulties": difficulties, 
            "labels": labels
            }
        return sample