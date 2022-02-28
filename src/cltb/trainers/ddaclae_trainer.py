from trainers.abstract_trainer import AbstractTrainer
from trainers.trainer_utils import encode_batch
from py_irt.scoring import calculate_theta
import numpy as np
import pandas as pd
import csv
import datetime

class DDaCLAETrainer(AbstractTrainer):
    """Class implementing the DDaCLAE algorithm for CL training"""
    def __init__(self, config):
        self.config = config
        self.num_epochs = self.config["trainer"]["num_epochs"]
        self.device = self.config["trainer"]["device"]
        self.batch_size = self.config["trainer"]["batch_size"]
        self.expname = self.config["trainer"]["expname"]

    
    def get_time(self):
        return str(datetime.datetime.now(datetime.timezone.utc))
    
    def get_difficulties(self, model, data, dev_data, e, outwriter):
        # I want this to look like the following:
        # first, try to look up the difficulties
        # second, learn difficulties

        # does the data difficulty exist?
        # try to read the difficulty file from config
        # CHANGE THIS:
        # DIFFICULTY SHOULD BE LOADED WHEN DATA IS LOADED
        # AND SET WHEN NEEDED DURING TRAINING
        self.difficulties_file = self.config["data"]["difficulties_file"]
        try:
            difficulties = pd.read_csv(self.difficulties_file, sep=',',
                                header=None, names=['id', 'difficulty'])
            difficulties = difficulties.set_index('id')
            difficulties = difficulties["difficulty"].to_dict()
            #print(difficulties)
            difficulties = [difficulties[int(i)] for i in data.ids]
            difficulties = np.array(difficulties)
        except:
            difficulties = self.learn_difficulties(model, data, e, outwriter)
        return difficulties
        
    def get_schedule(self, model,  data, dev_data, e, epoch_data_difficulties, outwriter):
        self.probe_set_size = self.config["data"]["probe_set_size"]
        #data["difficulties"] = epoch_data_difficulties
        theta_data = data.get_probe_set(self.probe_set_size)
        theta_hat = self.estimate_theta(model, theta_data) 
        outwriter.writerow([self.get_time(),e,"theta_hat",theta_hat])
        diffs = data.difficulties
        epoch_training_data = self.filter_examples(data, theta_hat, diffs) 
        return epoch_training_data

    def learn_difficulties(self, model, data, e, outwriter):
        raise NotImplementedError("TBD, stay tuned. For now learn difficulties offline via artificial crowd.")

    def estimate_theta(self, model, theta_data):
        model.model.eval()
        batch_size = self.config["trainer"]["batch_size"]
        all_preds = []

        for j in range(len(theta_data)//batch_size):
            batch_idx = [i for i in range(j*batch_size, min((j+1)*batch_size, len(theta_data["examples"])))]
            inputs, labels = encode_batch(theta_data, batch_idx, self.config, model)
            all_preds.extend(labels)
            inputs2 = {}
            for key, val in inputs.items():
                inputs2[key] = val.to(self.device)

            outputs = model.forward(inputs2)
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds) 
        
        rps = [int(p == c) for p, c in zip(all_preds, theta_data["labels"])] 
        rps = [j if j==1 else -1 for j in rps]
        diffs = theta_data["difficulties"]
        theta_hat = calculate_theta(diffs, rps)[0]
        return theta_hat

    def filter_examples(self, data, theta_hat, difficulties):
        idx = [i for i in range(len(difficulties)) if difficulties[i] <= theta_hat]
        return data[idx] 

"""
I'll delete the below once I'm sure I'm done with it
    def train(self):
        for e in range(self.num_epochs):
            # estimate theta
            theta_hat = self.estimate_theta() 
            self.outwriter.writerow([self.get_time(),e,"theta_hat",theta_hat])

            # filter out needed training examples from full set
            epoch_training_data = self.filter_examples(theta_hat) 
            self.outwriter.writerow([self.get_time(),e,"train_set_size",len(epoch_training_data["examples"])])

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
            self.outwriter.writerow([self.get_time(),e,"train_acc",acc])
            self.outwriter.writerow([self.get_time(),e,"train_loss",global_loss.cpu().detach().numpy()])

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
            self.outwriter.writerow([self.get_time(),e,"dev_acc",acc])

            self.outwriter.writerow([self.get_time(),e,"dev_loss",global_loss.cpu().detach().numpy()])

            # save model to disk if it's best performing so far 
            #self.model.save() 
        self.outwriter.writerow([self.get_time(),self.num_epochs,"endttime",self.get_time(),])
"""


