from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from abstract_model import AbstractModel

OPTIMIZERS = {
    "AdamW": AdamW
}

SCHEDULERS = {
    "constant": get_constant_schedule,
    "linear": get_linear_schedule_with_warmup
}

class BertModel(AbstractModel):
    def __init__(self, config):
        self.config = config
        self.device = self.config.trainer["device"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model["modelname"]
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model["modelname"],
            num_labels = self.config.data["num_labels"]
        )
        self.model.to(self.device) 
        self.optimizer = OPTIMIZERS[self.config.model["optimizer"]]
        self.scheduler = SCHEDULERS[self.config.model["scheduler"]] 


    def forward(self, epoch_training_data):
        logits = []
        for j, batch in enumerate(epoch_training_data):
            batch = tuple(t.to(self.device) for t in batch) 
            inputs = self.encode_batch(batch)
            #inputs = self.tokenizer(batch, return_tensors="pt")
            outputs = self.model(**inputs) 
            loss = outputs[0]
            logits.extend(outputs[1].detach().cpu().numpy())
            loss.backward() 
            self.optimizer.step() 
            self.scheduler.step()
            self.model.zero_grad()
        return logits

    def encode_batch(self, examples):
        if self.config.data["paired_inputs"]:
            batch_s_1 = examples.examples
            batch_s_2 = examples.examples2
            encoded_inputs = self.tokenizer(
                batch_s_1,
                batch_s_2,
                return_tensors="pt",
                max_length=self.config.trainer["max_seq_len"]
            )

        return encoded_inputs
