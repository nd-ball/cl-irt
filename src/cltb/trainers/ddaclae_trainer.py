from trainers.abstract_trainer import AbstractTrainer
from py_irt.scoring import calculate_theta
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, Dataset
import pandas as pd

class DDaCLAETrainer(AbstractTrainer):
    """Class implementing the DDaCLAE algorithm for CL training"""
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.probe_set_size = config["data"]["probe_set_size"]
        self.theta_data = self.data.get_probe_set(self.probe_set_size)
        self.num_epochs = config["trainer"]["num_epochs"]
        self.device = config["trainer"]["device"]
        self.batch_size = config["trainer"]["batch_size"]

        
    def train(self):
        """
        at each timestep:
        a) estimate model theta
        b) filter training data so that you only include those where b <= theta
        c) run a training pass 
        """

        for e in range(self.num_epochs):
            # estimate theta
            theta_hat = self.estimate_theta() 

            # filter out needed training examples from full set
            epoch_training_data = self.filter_examples(theta_hat) 

            # run one training step 
            # calculate loss and backprop 

            # training 
            self.model.model.train()
            loss, logits = self.model.forward(epoch_training_data) 
            # eval 
            #self.model.eval()

            # save model to disk if it's best performing so far 
            #self.model.save() 


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
        _, logits = self.model.forward(self.theta_data)
        preds = np.argmax(logits, axis=1) 
        
        rps = [int(p == c) for p, c in zip(preds, self.theta_data["labels"])] 
        rps = [j if j==1 else -1 for j in rps] 
        theta_hat = calculate_theta(self.theta_data["difficulties"], rps)[0]
        return theta_hat

    def filter_examples(self, theta_hat):
        idx = [i for i in range(len(self.data.difficulties)) if self.data.difficulties.difficulty[i] <= theta_hat]
        return self.data[idx] 





