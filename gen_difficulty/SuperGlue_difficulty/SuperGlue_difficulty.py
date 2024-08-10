import numpy as np
import random
import gc
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import transformers
import time
import os
import datetime
import torch
from transformers import AutoModel
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import json
from transformers import (GPT2ForSequenceClassification, GPT2Tokenizer,
                          DebertaForSequenceClassification, \
    ElectraTokenizerFast, ElectraForSequenceClassification,
                          T5ForSequenceClassification, T5Tokenizer)
from transformers import BartForSequenceClassification, BartTokenizer, DebertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW  # Import AdamW from torch.optim
from transformers import get_linear_schedule_with_warmup
import evaluate
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

os.environ["TOKENIZERS_PARALLELISM"] = "false"

models = [
"t5-base"
# "deberta-base",
#
#     "albert-base-v2",
#     "bert-base-uncased",
#     "bart-base",
#     "distilbert-base-uncased",
#
#     "gpt2",
#     "roberta-base",
#
#     "xlnet-base-cased",
#     "electra-base-discriminator"
]

SUPERGLUE_TASKS = ["cb", "copa", "rte", "wic", "wsc", "boolq"]

GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available: ", GPU_avail)


def tokenize_function(examples):
    if task == "cb":
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length",
                         max_length=max_length)
    elif task == "copa":
        return tokenizer(examples["premise"], examples["choice1"], examples["choice2"], truncation=True,
                         padding="max_length", max_length=max_length)
    elif task == "rte":
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length",
                         max_length=max_length)
    elif task == "wic":
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length",
                         max_length=max_length)
    elif task == "wsc":
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    elif task == "boolq":
        return tokenizer(examples["question"], examples["passage"], truncation=True, padding="max_length",
                         max_length=max_length)


train_batch_size = 4
dev_batch_size = 64

fine_tuning_epochs = [0, 1, 3, 5]
max_length = 512


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
    }

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#
#     if task == "cb":
#         metric = evaluate.load("super_glue", "cb", trust_remote_code=True)
#         results = metric.compute(predictions=predictions, references=labels)
#         return {"accuracy": results["accuracy"]}
#     elif task == "copa":
#         metric = evaluate.load("super_glue", "copa", trust_remote_code=True)
#         return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
#     elif task == "rte":
#         metric = evaluate.load("super_glue", "rte", trust_remote_code=True)
#         return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
#     elif task == "wic":
#         metric = evaluate.load("super_glue", "wic", trust_remote_code=True)
#         return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
#     elif task == "wsc":
#         metric = evaluate.load("super_glue", "wsc", trust_remote_code=True)
#         return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
#     elif task == "boolq":
#         metric = evaluate.load("super_glue", "boolq", trust_remote_code=True)
#         return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}


for task in SUPERGLUE_TASKS:
    print(f"\nTask: {task}")
    dataset = load_dataset("super_glue", task, trust_remote_code=True)
    print(f"Dataset Keys: {dataset.keys()}")

    train_dataset = dataset["validation"]
    pred_dataset = dataset["train"]

    train_size = len(train_dataset)
    pred_size = len(pred_dataset)

    print(f"Train size: {train_size}")
    print(f"Prediction size: {pred_size}")

    for model_checkpoint in models:
        # if model_checkpoint =="t5_base":
        #     train_batch_size = 8
        #     dev_batch_size = 8
        # else:
        #     train_batch_size = 16
        #     dev_batch_size = 16

        for num_epochs in fine_tuning_epochs:
            print("===========================================")
            print("Start test model --", model_checkpoint, " on task --", task, " with fine-tuning epochs --",
                  num_epochs)

            if model_checkpoint == "deberta-base":
                tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
                num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1
                model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base',
                                                                           num_labels=num_labels)
            elif model_checkpoint == "electra-base-discriminator":
                tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
                num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1
                model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator",
                                                                         num_labels=num_labels)
            elif model_checkpoint == "gpt2":
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1
                model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id
            elif model_checkpoint == "bart-base":
                tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
                num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1
                model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=num_labels)
            elif model_checkpoint == "t5-base":
                tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=max_length)
                num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1
                model = T5ForSequenceClassification.from_pretrained("t5-base", num_labels=num_labels)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
                num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1
                model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

            tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
            tokenized_pred_dataset = pred_dataset.map(tokenize_function, batched=True)

            tokenized_train_dataset.set_format('torch')
            tokenized_pred_dataset.set_format('torch')

            train_dataloader = DataLoader(tokenized_train_dataset, batch_size=train_batch_size, shuffle=True)
            pred_dataloader = DataLoader(tokenized_pred_dataset, batch_size=dev_batch_size, shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            model = model.to(device)

            if num_epochs > 0:
                model.train()
                optimizer = AdamW(model.parameters(), lr=5e-5)  # Use AdamW from torch.optim
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                            num_training_steps=len(train_dataloader) * num_epochs)
                time_s = time.time()

                for epoch in range(num_epochs):
                    print("Training Epoch:", epoch)
                    for batch in tqdm(train_dataloader):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)

                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs[0]
                        loss.backward()

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()

                time_e = time.time()
            else:
                time_s = time.time()
                time_e = time.time()

            print("\nRunning Test (on train set)...")
            model.eval()
            test_predictions = []
            test_labels = []

            for batch in tqdm(pred_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    test_predictions.append(logits.cpu().numpy())
                    test_labels.append(labels.cpu().numpy())

            test_predictions = np.concatenate(test_predictions, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            # test_metrics = compute_metrics((test_predictions, test_labels))
            # test_accuracy = test_metrics['accuracy']
            test_metrics = compute_metrics((test_predictions, test_labels))
            test_accuracy = test_metrics['accuracy']

            print(f"Test Accuracy: {test_accuracy:.4f}")

            train_time = time_e - time_s
            output_dir = f"finetuning_{num_epochs}_epochs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir,
                                    f"{model_checkpoint}_{task}_{train_time}_response_logits_Accuracy_{test_accuracy}.json")
            data = {
                "logits": test_predictions.tolist(),
                "responses": test_labels.tolist()
            }

            with open(filename, "w") as f:
                json.dump(data, f)
