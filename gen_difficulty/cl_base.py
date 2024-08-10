import numpy as np
import argparse
import random
import gc
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import transformers
# from torch.optim import AdamW
import time
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import datetime
import time
from transformers import AutoModel
#from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import required bert libraries
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer,DebertaV2ForSequenceClassification, ElectraTokenizerFast,ElectraForSequenceClassification
from transformers import BartForSequenceClassification, BartTokenizer,DebertaV2Tokenizer
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForSequenceClassification, BertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.data.processors import utils
from transformers import glue_convert_examples_to_features as convert_examples_to_features

models = [
    #
    "bart-base",
    "roberta-base",
    "albert-base-v2",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "deberta-base",
    "gpt2",
    "t5-base",
    "xlnet-base-cased",
    "electra-base-discriminator"
           ]
#
# GLUE_TASKS = ["cola", "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
GLUE_TASKS = [ "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2"]
# GLUE_TASKS = [  "rte", "sst2"]
GPU_avail= torch.cuda.is_available()
print("GPU_CUDA is available: ",GPU_avail)


def tokenize_function(examples):
    if task in ["mnli"]:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", truncation=True, max_length=max_length)
    elif task in ["mrpc", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         padding="max_length", truncation=True, max_length=max_length)
    elif task in ["qnli"]:
        return tokenizer(examples["question"], examples["sentence"],
                         padding="max_length", truncation=True, max_length=max_length)
    elif task in ["qqp"]:
        return tokenizer(examples["question1"], examples["question2"],
                         padding="max_length", truncation=True, max_length=max_length)

    elif task in ["sst2"]:
        return tokenizer(examples["sentence"],
                         padding="max_length", truncation=True, max_length=max_length)


train_batch_size =32
dev_batch_size = 32

fine_tuning=True
num_epochs=10
max_length=128
for task in GLUE_TASKS:
    print("Task:", task)
    dataset = load_dataset("glue", task)
    print(dataset.keys())
    for model_checkpoint in models:
        print("===========================================")
        print("Start test model --", model_checkpoint, " on task--", task)
        if model_checkpoint == "deberta-v3-large":
            tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-large', num_labels=num_labels)


        elif model_checkpoint == "electra-base-discriminator":
            tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator", num_labels=num_labels)
            # batch_size = 64
        elif model_checkpoint =="gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        elif model_checkpoint == "bart-base":
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            model = BartForSequenceClassification.from_pretrained("facebook/bart-base",num_labels=num_labels)
            # batch_size = 64
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=train_batch_size, shuffle=True)
        if fine_tuning ==True:
            if task in ["mnli"]:
                dev_dataloader = DataLoader(tokenized_dataset["validation_matched"], batch_size=dev_batch_size, shuffle=True)
                pred_dataloader = DataLoader(tokenized_dataset["test_matched"], batch_size=dev_batch_size, shuffle=True)
            if task in ["mrpc","qnli", "qqp", "rte", "sst2"]:
                dev_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=dev_batch_size, shuffle=True)
                pred_dataloader = DataLoader(tokenized_dataset["test"], batch_size=dev_batch_size, shuffle=True)
        # predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # Move the model to the device
        model = model.to(device)
        # fine-tuning on the dev dataset
        if fine_tuning == True:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10,
                                                        num_training_steps=len(train_dataloader) * num_epochs)
            time_s=time.time()
            for epoch in range(num_epochs):
                print("Training Epoch:", epoch)
                for batch in tqdm(train_dataloader):
                    # Move batch to the appropriate device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs[0]
                    loss.backward()

                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()

        time_e=time.time()
        model.eval()
        predictions = []
        logits_list = []
        responses = []
        print("Start to calculate the accuracy:")
        for batch in tqdm(pred_dataloader):
            # Move batch to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs[1]
                logits = torch.nn.functional.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()

                out_label_ids = batch['label'].detach().cpu().numpy()


                res=np.equal(pred, out_label_ids).astype(int)
                logits_list.append(logits.detach().cpu().numpy())
                responses.append(res)


        print()
        responses_arr=np.concatenate(responses)
        logits_arr=np.concatenate(logits_list,axis=0)
        # print("Task:", task)
        train_time=time_e-time_s
        print("Accuracy is :", responses_arr.mean())
        print("Traing time:", train_time)
        filename = f"{model_checkpoint}_{task}_{train_time}_response_logits_Accuracy_{responses_arr.mean()}_finetuning_{num_epochs}_epochs.json"

        data = {
            "logits": logits_arr.tolist(),
            "responses": responses_arr.tolist()
        }

        with open(filename, "w") as f:
            json.dump(data, f)




