from datasets.cl_dataset import CLDataset 
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets.utils import download_url, check_integrity
import os
from torchvision import transforms
import csv
from PIL import Image

FNAMES = {
    "train": "snli_1.0_train.txt",
    "dev": "snli_1.0_dev.txt",
    "test": "snli_1.0_test.txt"
}


class MNISTDataset(CLDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}
    def __init__(self,config):
        self.split = config["data_split"]
        self.data_path = config["data_path"]
        #self.data_path = f"{self.data_path}/{FNAMES[self.split]}"

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

        self.root = os.path.expanduser(self.data_path)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.target_transform = None
        self.train =  self.split == "train" # training set or test set

        if not self._check_exists():
            self.download()
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, self.processed_folder, data_file))

        self.idx = torch.tensor(list(range(len(self.data))), dtype=torch.int)

        self.difficulties = []
        diff_dir = config["difficulties_file"] 
        if config["difficulties_file"]:            
            if self.train:
                diff_file = diff_dir + 'mnist_diffs_train.csv'
            else:
                diff_file = diff_dir + 'mnist_diffs_test.csv' 
            with open(diff_file, 'r') as infile:
                diffreader = csv.reader(infile, delimiter=',')

                for row in diffreader:
                    self.difficulties.append((eval(row[0]), eval(row[1])))
                self.difficulties_sorted = sorted(self.difficulties, key=lambda x:x[0])
                self.difficulties = {}
                for key, val in self.difficulties_sorted:
                    self.difficulties[key] = val
        else:
            self.difficulties = {}
            for i in range(len(self.data)):
                self.difficulties[i] = 0
        
       

    def __getitem__(self, idx):
        img, target, label, diff = self.data[idx], self.targets[idx], self.idx[idx], self.difficulties[idx]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = {
            "ids": label,
            "examples": img,
            "examples2": None,  
            "difficulties": diff, 
            "labels": target
            }
        return sample

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, label, diff = self.data[index], self.targets[index], self.idx[index], self.difficulties[index]
        pcorrect = self.pcorrect[index] 

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)



        return img, target, label, diff, pcorrect



def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)