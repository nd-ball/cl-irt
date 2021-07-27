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


    def forward(self, epoch_training_data):
        logits = []
        global_loss = 0
        batch_size = self.config["trainer"]["batch_size"]
        for j in range(len(epoch_training_data)//batch_size):
            batch_idx = [i for i in range(j*batch_size, min((j+1)*batch_size, len(epoch_training_data)))]
            inputs, labels = self.encode_batch(epoch_training_data, batch_idx)
            print(inputs)
            inputs2 = {}
            for key, val in inputs.items():
                inputs2[key] = val.to(self.device)

            outputs = self.model(**inputs2, labels=labels)
            loss = outputs.loss
            logits.extend(outputs.logits.detach().cpu().numpy())
            loss.backward() 
            self.optimizer.step() 
            self.scheduler.step()
            self.model.zero_grad()
            global_loss += loss
        return global_loss, logits

    def encode_batch(self, examples, batch_idx):
        if self.config["data"]["paired_inputs"]:
            batch_s_1 = [examples["examples"][i] for i in batch_idx]
            batch_s_2 = [examples["examples2"][i] for i in batch_idx]
            encoded_inputs = self.tokenizer(
                batch_s_1,
                batch_s_2,
                return_tensors="pt",
                max_length=self.config["trainer"]["max_seq_len"],
                padding=True,
                truncation=True
            )
        else:
            batch_s_1 = [examples["examples"][i] for i in batch_idx]
            encoded_inputs = self.tokenizer(
                batch_s_1,
                return_tensors="pt",
                max_length=self.config["trainer"]["max_seq_len"],
                padding=True,
                truncation=True
            )
        batch_labels = [examples["labels"][i] for i in batch_idx]
        batch_labels = torch.tensor(batch_labels, device=self.device)

        return encoded_inputs, batch_labels
