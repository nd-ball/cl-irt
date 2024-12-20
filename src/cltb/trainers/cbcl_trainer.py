"""
Competency-based curriculum learning trainer
"""

from trainers.abstract_trainer import AbstractTrainer
import numpy as np
import pandas as pd
import torch
import csv
import datetime

class CBCLTrainer(AbstractTrainer):
    """Class implementing the CBCL algorithm for CL training"""
    def __init__(self, config):
        self.config = config
        self.probe_set_size = self.config["data"]["probe_set_size"]
        self.theta_data = self.data.get_probe_set(self.probe_set_size)
        self.num_epochs = self.config["trainer"]["num_epochs"]
        self.device = self.config["trainer"]["device"]
        self.batch_size = self.config["trainer"]["batch_size"]
        self.expname = self.config["trainer"]["expname"]
        self.outwriter = csv.writer(open(f"logs/{self.expname}.log", "w"))

        # CBCL
        self.competencyfn = self.config["trainer"]["competencyfn"]
        self.c_init = self.config["trainer"]["c_init"]
        self.competency = self.config["trainer"]["competency"]
    
    def get_time(self):
        return str(datetime.datetime.now(datetime.timezone.utc))

    def get_difficulties(self, model, data, dev_data, e, outwriter):
        # I want this to look like the following:
        # first, try to look up the difficulties
        # second, learn difficulties

        # does the data difficulty exist?
        # try to read the difficulty file from config
        self.difficulties_file = self.config["data"]["difficulties_file"]
        try:
            difficulties = pd.read_csv(self.difficulties_file, sep=',',
                                header=None, names=['id', 'difficulty'])
        except:
            difficulties = self.learn_difficulties(model, data, e, outwriter)
        return difficulties

    def get_schedule(self, model,  data, dev_data, e, epoch_data_difficulties, outwriter):
        epoch_training_data = self.filter_examples(data, e, epoch_data_difficulties) 
        return epoch_training_data

    def filter_examples(self, data, e, epoch_data_difficulties):
        # sorted difficulty indices
        diffs_sorted_idx = np.argsort(epoch_data_difficulties) 

        # calculate competency
        if self.competencyfn == "linear":
            epoch_competency = np.min([1, e * ((1 - self.c_init)/self.competency) + self.c_init])
        elif self.competencyfn == "root":
            epoch_competency = np.min([1,np.sqrt(e * ((1 - self.c_init**2)/self.competency) + self.c_init**2)])
        else:
            raise NotImplementedError("other competency functions not implemented.")

        # get indices
        idx = [diffs_sorted_idx[i] for i in range(epoch_competency * len(self.data.difficulties))]
        epoch_training_data = data[idx]  
        return epoch_training_data

"""
Will delete once I know I don't need it
    def train(self):
        ""
        at each timestep:
        a) estimate model theta
        b) filter training data so that you only include those where b <= theta
        c) run a training pass 
        ""
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

            # filter out needed training examples from full set
            epoch_training_data = self.filter_examples(e) 
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
"""

