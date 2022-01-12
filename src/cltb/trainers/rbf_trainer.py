"""spaced repitition trainer"""

from trainers.abstract_trainer import AbstractTrainer
from trainers.trainer_utils import calculate_accuracy, encode_batch
from py_irt.scoring import calculate_theta
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import pandas as pd
import torch
import csv
import datetime

class RbFTrainer(AbstractTrainer):
    """Class implementing the RbF algorithm for CL training"""
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
        self.nu = 0.5
        self.kern = self.config["trainer"]["kern"]
    
    def get_time(self):
        return str(datetime.datetime.now(datetime.timezone.utc))

    def train(self):
        """
        in this model difficulty is effectively (next epoch to train at)

        at each timestep:
        
        a) select training examples where diff = current_epoch
        b) train
        c) calulate tao on validation data (expensive?)
        d) estimate tao hat for each trained examples
        e) update diffs accordingly
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

        self.data.difficulties.difficulty = [0 for i in range(len(self.data.difficulties.difficulty))]
        
        for e in range(self.num_epochs):
            # filter out needed training examples from full set
            idx = self.filter_examples(e)
            print(len(idx))
            if len(idx) < 300:
                print("too few for training")
                continue
            epoch_training_data = self.data[idx]
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

            for j in range(len(epoch_training_data["examples"])//batch_size + 1):
                batch_idx = [i for i in range(j*batch_size, min((j+1)*batch_size, len(epoch_training_data["examples"])))]
                if len(batch_idx) == 0:
                    continue
                inputs, labels = encode_batch(epoch_training_data, batch_idx)
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
                global_loss += loss

            acc = calculate_accuracy(logits, all_labels)
            all_labels = [l.cpu().numpy() for l in all_labels]
            train_accuracies = np.argmax(logits, axis=1) == all_labels
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
                inputs, labels = encode_batch(epoch_dev_data, batch_idx)
                all_labels.extend(labels)

                inputs2 = {}
                for key, val in inputs.items():
                    inputs2[key] = val.to(self.device)

                with torch.no_grad():
                  outputs = self.model.forward(inputs2, labels)
                  loss = outputs.loss
                  self.model.model.zero_grad()

                logits.extend(outputs.logits.detach().cpu().numpy())
                global_loss += loss
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

            all_labels = [l.cpu().numpy() for l in all_labels]
            val_accuracies = (np.argmax(logits, axis=1) == all_labels) * 1
            val_loss = [-1 * logits[i][all_labels[i]] for i in range(len(val_accuracies))]

            optimal_tau = self.get_optimal_tau_rbf(self.kern, val_accuracies, val_loss, self.nu)

            new_delays = self.calculate_delays(optimal_tau, train_accuracies, self.nu, self.kern)
            assert len(new_delays) == len(idx)
            for i in range(len(idx)):
                self.data.difficulties.difficulty[idx[i]] += new_delays[i]
                # we want a full pass through the data at the last epoch
                if self.data.difficulties.difficulty[idx[i]] >= self.num_epochs:
                    self.data.difficulties.difficulty[idx[i]] = self.num_epochs - 1


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

        
    def filter_examples(self, e):
        idx = [i for i in range(len(self.data.difficulties.difficulty)) if self.data.difficulties.difficulty[i] <= e]
        return idx


    def get_optimal_tau_rbf(self, kern, val_accs, val_loss, nu):
        x = 1.0 / np.sum(val_accs)
        
        if kern == 'gau':
            a_ln = -1. * np.sum([np.log(a- 0.1) for a in val_accs if a >= nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(val_loss, val_accs) if a >= nu])
            tau = a_ln / x_sum_pow
        
        if kern == 'lap':
            a_ln = -1. * np.sum([np.log(a) for a in val_accs if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(val_loss, val_accs) if a >= nu])                        
            tau = a_ln / x_sum
        
        if kern == 'lin':
            a_one = np.sum([(1. - a) for a in val_accs if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(val_loss, val_accs) if a >= nu])                            
            tau = a_one / x_sum
        
        if kern == 'cos':
            a_arc = np.sum([np.arccos(2. * a - 1.) for a in val_accs if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(val_loss, val_accs) if a >= nu])
            tau = a_arc / (np.pi * x_sum)
        
        if kern == 'qua':
            a_one = np.sum([(1. - a) for a in val_accs if a >= nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(val_loss, val_accs) if a >= nu])           
            tau = a_one / x_sum_pow
        
        if kern == 'sec':
            a_sq = np.sum([np.log(1. / a + np.sqrt(1. / a - 1.)) for a in val_accs if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(val_loss, val_accs) if a >= nu])                            
            tau = a_sq / x_sum
        
        return tau 

    def calculate_delays(self, tau, accuracies, nu, kern):
        delays = []
        val_strength = np.sum(accuracies)
        if kern == "gau":
            nu_gau = np.sqrt(-np.log(nu) / tau)
            fn = lambda a : max(1., val_strength * nu_gau / a)
        if kern == "lap":
            nu_lap = np.log(nu)    
            fn = lambda a :  max(1., -1. * val_strength * nu_lap / (a * tau))
        if kern == "lin":
            nu_lin = (1. - nu)
            fn = lambda a : max(1., val_strength * nu_lin / (a * tau))
        
        if kern == 'cos':
            nu_cos = np.arccos(2 * nu - 1.)
            fn = lambda a :  max(1., val_strength * nu_cos / (np.pi * a * tau))
                        
        if kern == 'qua':
            nu_qua = np.sqrt((1. - nu) / tau)
            fn = lambda a :  max(1., val_strength * nu_qua / a)
                            
        if kern == 'sec':
            nu_sec = np.log(1. / nu * (1 + np.sqrt(1 - nu * nu)))
            fn = lambda a :  max(1., val_strength * nu_sec / (a * tau))
            
        for i in range(len(accuracies)):
            if accuracies[i] < nu:
                delays.append(1)
            else:
                delays.append(min(fn(accuracies[i]), self.num_epochs-1))
        return delays
                        
        
