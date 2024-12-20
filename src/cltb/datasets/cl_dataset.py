"""
Datasets will be implemented as subclasses of the CLDataset class (maybe I rename this)?

CLDataset will subclass pytorch datasets, but require difficulty values for entries 

We're making the following assumptions regarding the datasets

1. every entry  has a unique identifier
2. difficulty files are stored in the format ID, difficulty
"""

from torch.utils.data import Dataset
from typing import List, Optional
import random
import numpy as np


class CLDataset(Dataset):
    """class for CL datasets"""
    ids: List[int]
    difficulties: Optional[List[float]] = None
    examples = List[str]
    labels = List[int]

    def __init__(self, config):
        pass 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        pass

    def fit_latent_crowd(self):
        pass
    
    def get_probe_set(self, num_items):
        """return a random sample from the data set for estimating model ability"""
        idx = random.sample(range(0, len(self.examples)), num_items)
        return self.__getitem__(idx)


