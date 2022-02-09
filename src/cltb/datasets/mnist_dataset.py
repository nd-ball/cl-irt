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
import codecs
import errno

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
    def __init__(self,config,mode="train"):
        self.split = config["data"]["data_split"]
        self.data_path = config["data"]["data_path"]
        #self.data_path = f"{self.data_path}/{FNAMES[self.split]}"

        self.root = os.path.expanduser(self.data_path)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.target_transform = None
        self.train =  mode == "train" # training set or test set

        if not self._check_exists():
            self.download()
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, self.processed_folder, data_file))

        self.idx = torch.tensor(list(range(len(self.data))), dtype=torch.int)

        self.difficulties = []
        diff_dir = config["data"]["difficulties_file"] 
        if config["data"]["difficulties_file"]:            
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

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            download_url(url, root=os.path.join(self.root, self.raw_folder),
                         filename=filename, md5=None)
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
        
       

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

