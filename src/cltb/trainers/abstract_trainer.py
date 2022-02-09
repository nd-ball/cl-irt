"""
Trainers will be built off of an abstract CLTrainer class

They will need to:

call trainer.train(model, data)

where are hyperparams held? at the initialization of the trainer

myTrainer = Trainer(args)  
"""


import abc 
from typing import Dict, Any


class AbstractTrainer(abc.ABC):
    def __init__(self, *args):
        pass    
    


