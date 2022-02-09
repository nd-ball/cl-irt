from trainers.abstract_trainer import AbstractTrainer
from trainers.trainer_utils import encode_batch
from py_irt.scoring import calculate_theta
import numpy as np
import pandas as pd
import csv
import datetime

class HacohenTrainer(AbstractTrainer):
    """Class implementing the Hacohen algorithm for CL training"""
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
        difficulties = "TODO"
        return difficulties
        
    def get_schedule(self, model,  data, dev_data, e, epoch_data_difficulties, outwriter):
        epoch_training_data = "TODO"
        return epoch_training_data

    def learn_difficulties(self, model, data, e, outwriter):
        raise NotImplementedError("TBD, stay tuned. For now learn difficulties offline via artificial crowd.")

    