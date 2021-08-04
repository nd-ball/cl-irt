from datasets.cl_dataset import CLDataset 
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder



class GLUEDataset(CLDataset):
    def __init__(self,config,mode="train"):
        '''
        load and return the glue data with difficulties
        '''
        self.taskname = config["data"]["taskname"]
        GLUETASKS = ['CoLA', 'SST-2', 'MRPC', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'QQP']

        assert self.taskname in GLUETASKS, 'task not found' 


        self.split = mode #config["data"]["data_split"]
        self.data_path = config["data"]["data_path"]
        self.data_file = f"{self.data_path}/{self.taskname}/{self.split}.tsv"
        self.ids = []
        self.examples = []
        self.paired_inputs = config["data"]["paired_inputs"]
        if self.paired_inputs:
            self.examples2 = []
        self.str_labels = []


        with open(self.data_file, 'r') as infile:
            idx = 0
            if self.taskname != 'CoLA': 
                next(infile) 
            for line in infile:
                split_line = line.strip().split('\t') 
                lid, s1, s2, label = self.parse_line(split_line, self.taskname, self.split)
                if lid == -1:
                    lid = idx 
                idx += 1 
                self.examples.append(s1)
                if self.paired_inputs:
                    self.examples2.append(s2) 
                self.str_labels.append(label) 
                self.ids.append(lid) 

        le = LabelEncoder()
        self.labels = le.fit_transform(self.str_labels)

        # load difficulties 
        if config["data"]["difficulties_file"]:
            self.difficulties_file = config["data"]["difficulties_file"]
            self.difficulties = pd.read_csv(self.difficulties_file, sep=',',
                                header=None, names=['id', 'difficulty'])



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        examples = [self.examples[i] for i in idx]
        examples = np.array(examples)
        if self.paired_inputs:
            examples2 = [self.examples2[i] for i in idx]
            examples2 = np.array(examples2)
        difficulties = [self.difficulties.difficulty[i] for i in idx]
        difficulties = np.array(difficulties)
        labels = [self.labels[i] for i in idx]
        labels = np.array(labels)
        if self.paired_inputs:
            sample = {
                "examples": examples,
                "examples2": examples2,  
                "difficulties": difficulties, 
                "labels": labels
            }
        else:
            sample = {
                "examples": examples,
                "difficulties": difficulties, 
                "labels": labels
            }
        return sample


    def parse_line(self, line, task, split):
        '''
        1: train: return id, s1, s2, label
        2: dev: return id, s1, s2, label
        3: test: return id, s1, s2
        '''
        if task == 'CoLA':
            if split in ["train", "dev"]:
                return line[0], line[3], np.NaN, line[1] 
            else:
                return line[0], line[1], np.NaN
        elif task == 'SST-2':
            if split in ["train", "dev"]:
                return -1, line[0], np.NaN, line[1] 
            else:
                return line[0], line[1], np.NaN
        elif task == 'MRPC':
            if split in ["train", "dev"]:
                return -1, line[3], line[4], line[0] 
            else:
                return line[0], line[3], line[4]
        elif task in ['QNLI', 'RTE', 'WNLI']:
            if split in ["train", "dev"]:
                return line[0], line[1], line[2], line[3] 
            else:
                return line[0], line[1], line[2]
        elif task == 'QQP':
            if split in ["train", "dev"]:
                return line[0], line[3], line[4], line[5] 
            else:
                return line[0], line[1], line[2]
        elif task == 'MNLI':
            if split == "train":
                return line[0], line[8], line[9], line[11]
            elif split == "dev-matched":
                return line[0], line[8], line[9], line[15]
            else:
                return line[0], line[8], line[9]
        else:
            raise NotImplementedError 
    