from datasets.cl_dataset import CLDataset 
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets.utils import download_url, check_integrity
import os
from torchvision import transforms
import csv

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
        
        self.transform = transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        self.target_transform = transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
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


class my_MNIST(data.Dataset):
    

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, diff_dir=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, self.processed_folder, data_file))

        self.idx = torch.tensor(list(range(len(self.data))), dtype=torch.int)

        self.difficulties = []
        if diff_dir is not None:
            if self.train:
                diff_file = diff_dir + 'mnist_diffs_train.csv'
            else:
                diff_file = diff_dir + 'mnist_diffs_test.csv' 
            with open(diff_file, 'r') as infile:
                diffreader = csv.reader(infile, delimiter=',')

                for row in diffreader:
                    self.difficulties.append((eval(row[0]), eval(row[1])))
                self.difficulties_sorted = sorted(self.difficulties, key=lambda x:x[0])
                self.difficulties = [d[1] for d in self.difficulties_sorted]

            self.difficulties = torch.tensor(self.difficulties, dtype=torch.float)
        else:
            self.difficulties = torch.tensor(np.zeros(len(self.data)), dtype=torch.float)
        
        if diff_dir is not None:
            self.pcorrect = []
            self.pcorrect_dict = {}
            if self.train:
                pcorrect_file = diff_dir + 'mnist_rp_train.csv'
            else:
                pcorrect_file = diff_dir + 'mnist_rp_test.csv'
            with open(pcorrect_file, 'r') as infile:
                pcorrect_reader = csv.reader(infile, delimiter=',')

                # trainsize,noise,itemid,response
                next(pcorrect_reader)
                for _, _, itemID, response in pcorrect_reader:
                    itemID = eval(itemID)
                    response = eval(response)
                    if itemID not in self.pcorrect_dict:
                        self.pcorrect_dict[itemID] = {
                            'correct': 0.0,
                            'total': 0.0
                        }
                    self.pcorrect_dict[itemID]['correct'] += response
                    self.pcorrect_dict[itemID]['total'] += 1
            self.pcorrect = [0.0] * len(self.idx)  
            for key, val in self.pcorrect_dict.items():
                self.pcorrect[key] = val['correct'] / val['total']
            self.pcorrect = torch.tensor(self.pcorrect, dtype=torch.float) 
        else:
            self.pcorrect = torch.zeros(len(self.idx), dtype=torch.float)
       

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.data)

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

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

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