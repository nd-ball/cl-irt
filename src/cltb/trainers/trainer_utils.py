import numpy as np
import torch

def calculate_accuracy(self, logits, labels):
        #print(logits)
        labels = [l.cpu().numpy() for l in labels]
        #print(labels)
        return np.sum(np.argmax(logits, axis=1) == labels) / len(logits)


def encode_batch(self, examples, batch_idx):
    if self.config["data"]["paired_inputs"]:
        batch_s_1 = [examples["examples"][i] for i in batch_idx]
        batch_s_2 = [examples["examples2"][i] for i in batch_idx]
        encoded_inputs = self.model.tokenizer(
            batch_s_1,
            batch_s_2,
            return_tensors="pt",
            max_length=self.config["trainer"]["max_seq_len"],
            padding=True,
            truncation=True
        )
    else:
        batch_s_1 = [examples["examples"][i] for i in batch_idx]
        encoded_inputs = self.model.tokenizer(
            batch_s_1,
            return_tensors="pt",
            max_length=self.config["trainer"]["max_seq_len"],
            padding=True,
            truncation=True
        )
    batch_labels = [examples["labels"][i] for i in batch_idx]
    batch_labels = torch.tensor(batch_labels, device=self.device)

    return encoded_inputs, batch_labels