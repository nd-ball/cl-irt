from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import torch
from models.abstract_model import AbstractModel

OPTIMIZERS = {
    "AdamW": AdamW
}

SCHEDULERS = {
    "constant": get_constant_schedule,
    "linear": get_linear_schedule_with_warmup
}

class BERTModel(AbstractModel):
    def __init__(self, config):
        self.config = config
        self.device = self.config["trainer"]["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["modelname"]
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model"]["modelname"],
            num_labels = self.config["data"]["num_labels"]
        )
        self.model.to(self.device) 
        self.optimizer = OPTIMIZERS[self.config["model"]["optimizer"]](
            self.model.parameters(), lr=2e-5, eps=1e-8, correct_bias=False
        )
        self.scheduler = SCHEDULERS[self.config["model"]["scheduler"]](self.optimizer)


    def forward(self, inputs2, labels):
        if labels is not None:
            outputs = self.model(**inputs2, labels=labels)
        else:
            outputs = self.model(**inputs2)
        return outputs

    
