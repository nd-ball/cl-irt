import numpy as np
import torch

def calculate_accuracy(logits, labels):
        #print(logits)
        labels = [l.cpu().numpy() for l in labels]
        #print(labels)
        return np.sum(np.argmax(logits, axis=1) == labels) / len(logits)


def encode_batch(examples, batch_idx, config, model):
    device = config["trainer"]["device"]
    if config["data"]["paired_inputs"]:
        batch_s_1 = [examples["examples"][i] for i in batch_idx]
        batch_s_2 = [examples["examples2"][i] for i in batch_idx]
        encoded_inputs = model.tokenizer(
            batch_s_1,
            batch_s_2,
            return_tensors="pt",
            max_length=config["trainer"]["max_seq_len"],
            padding=True,
            truncation=True
        )
    else:
        batch_s_1 = [examples["examples"][i] for i in batch_idx]
        encoded_inputs = model.tokenizer(
            batch_s_1,
            return_tensors="pt",
            max_length=config["trainer"]["max_seq_len"],
            padding=True,
            truncation=True
        )
    batch_labels = [examples["labels"][i] for i in batch_idx]
    batch_labels = torch.tensor(batch_labels, device=device)

    return encoded_inputs, batch_labels
