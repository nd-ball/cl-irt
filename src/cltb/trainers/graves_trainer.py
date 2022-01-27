from trainers.abstract_trainer import AbstractTrainer
from trainers.trainer_utils import calculate_accuracy, encode_batch
from py_irt.scoring import calculate_theta
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import pandas as pd
import torch
import csv
import datetime

class GravesTrainer(AbstractTrainer):
    """Class implementing the Graves bandit algorithm for CL training"""
    def __init__(self, model, data, dev_data, config):
        self.model = model
        self.data = data
        self.dev_data = dev_data
        self.config = config
        self.probe_set_size = self.config["data"]["probe_set_size"]
        self.theta_data = self.data.get_probe_set(self.probe_set_size)
        self.num_epochs = self.config["trainer"]["num_epochs"]
        self.device = self.config["trainer"]["device"]
        self.batch_size = self.config["trainer"]["batch_size"]
        self.expname = self.config["trainer"]["expname"]
        self.outwriter = csv.writer(open(f"logs/{self.expname}.log", "w"))

    
    def get_time(self):
        return str(datetime.datetime.now(datetime.timezone.utc))

    def train(self):
        """
        at each timestep:
        a) estimate model theta
        b) filter training data so that you only include those where b <= theta
        c) run a training pass 
        """
        # initialize logger
        self.outwriter.writerow(["timestamp","epoch","metric","value"])
        self.outwriter.writerow(
                [
                    self.get_time(),
                    -1,
                    "starttime",
                    self.get_time()
                ]
            )

        for e in range(self.num_epochs):
            # estimate theta
            theta_hat = self.estimate_theta() 
            self.outwriter.writerow(
                [
                    self.get_time(),
                    e,
                    "theta_hat",
                    theta_hat
                ]
            )

            # filter out needed training examples from full set
            epoch_training_data = self.filter_examples(theta_hat) 
            self.outwriter.writerow(
                [
                    self.get_time(),
                    e,
                    "train_set_size",
                    len(epoch_training_data["examples"])
                ]
            )


            # run one training step 
            # calculate loss and backprop 

            # training 
            self.model.model.train()
            logits = []
            all_labels = []
            global_loss = 0
            batch_size = self.config["trainer"]["batch_size"]

            for j in range(len(epoch_training_data["examples"])//batch_size):
                batch_idx = [i for i in range(j*batch_size, min((j+1)*batch_size, len(epoch_training_data["examples"])))]
                inputs, labels = encode_batch(epoch_training_data, batch_idx, self.config, self.model)
                all_labels.extend(labels)
                inputs2 = {}
                for key, val in inputs.items():
                    inputs2[key] = val.to(self.device)

                outputs = self.model.forward(inputs2, labels)
                loss = outputs.loss

                logits.extend(outputs.logits.detach().cpu().numpy())
                loss.backward() 
                self.model.optimizer.step() 
                self.model.scheduler.step()
                self.model.model.zero_grad()
                global_loss += loss.item()
            acc = calculate_accuracy(logits, all_labels)
            self.outwriter.writerow(
                [
                    self.get_time(),
                    e,
                    "train_acc",
                    acc
                ]
            )

            self.outwriter.writerow(
                [
                    self.get_time(),
                    e,
                    "train_loss",
                    global_loss.cpu().detach().numpy()
                ]
            )


            # eval 
            self.model.model.eval()
            logits = []
            all_labels = []
            global_loss = 0

            batch_size = self.config["trainer"]["batch_size"]
            dev_idx = list(range(len(self.dev_data.examples)))
            epoch_dev_data = self.dev_data[dev_idx] 
            for j in range(len(epoch_dev_data["examples"])//batch_size):
                batch_idx = [i for i in range(j*batch_size, min((j+1)*batch_size, len(epoch_dev_data["examples"])))]
                #batch_dev = [self.dev_data[i] for i in batch_idx]
                inputs, labels = encode_batch(epoch_dev_data, batch_idx, self.config, self.model)
                all_labels.extend(labels)

                inputs2 = {}
                for key, val in inputs.items():
                    inputs2[key] = val.to(self.device)


                outputs = self.model.forward(inputs2, labels)
                loss = outputs.loss
                self.model.model.zero_grad()

                logits.extend(outputs.logits.detach().cpu().numpy())
                global_loss += loss.item()
            acc = calculate_accuracy(logits, all_labels)
            self.outwriter.writerow(
                [
                    self.get_time(),
                    e,
                    "dev_acc",
                    acc
                ]
            )

            self.outwriter.writerow(
                [
                    self.get_time(),
                    e,
                    "dev_loss",
                    global_loss.cpu().detach().numpy()
                ]
            )


            # save model to disk if it's best performing so far 
            #self.model.save() 
        self.outwriter.writerow(
                [
                    self.get_time(),
                    self.num_epochs,
                    "endttime",
                    self.get_time(),
                ]
            )


    def estimate_theta(self):
        
        #theta_data = pd.DataFrame.from_dict(self.theta_data)
        """
        theta_sampler = SequentialSampler(
            theta_data
        )
        theta_dataloader = DataLoader(
            theta_data, 
            sampler=theta_sampler, 
            batch_size=self.batch_size
        )
        """
        self.model.model.eval()
        batch_size = self.config["trainer"]["batch_size"]
        all_preds = []

        for j in range(len(self.theta_data)//batch_size):
            batch_idx = [i for i in range(j*batch_size, min((j+1)*batch_size, len(self.theta_data["examples"])))]
            inputs, labels = encode_batch(self.theta_data, batch_idx, self.config, self.model)
            all_preds.extend(labels)
            inputs2 = {}
            for key, val in inputs.items():
                inputs2[key] = val.to(self.device)

            outputs = self.model.forward(inputs2)
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds) 
        
        rps = [int(p == c) for p, c in zip(all_preds, self.theta_data["labels"])] 
        rps = [j if j==1 else -1 for j in rps] 
        theta_hat = calculate_theta(self.theta_data["difficulties"], rps)[0]
        return theta_hat

    def filter_examples(self, theta_hat):
        idx = [i for i in range(len(self.data.difficulties)) if self.data.difficulties.difficulty[i] <= theta_hat]
        return self.data[idx] 


    


