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
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import datetime
import time
from transformers import AutoModel
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import required bert libraries
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, ElectraTokenizerFast,ElectraForSequenceClassification
from transformers import BartForSequenceClassification, BartTokenizer
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import (WEIGHTS_NAME, BertConfig,
                            BertForSequenceClassification, BertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.data.processors import utils
from transformers import glue_convert_examples_to_features as convert_examples_to_features

models = [
    "bert-base-uncased",
          "bert-base-cased",
          "roberta-base",
         'roberta-large',

          "albert-base-v2",
          "xlnet-base-cased",
          "electra-base-discriminator",
          "t5-base",

          "bart-base",
           "gpt2"]
#
# GLUE_TASKS = ["cola", "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
GLUE_TASKS = [ "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2"]

batch_size =8
max_length=256
for model_checkpoint in models:
    for task in GLUE_TASKS:
        dataset = load_dataset("glue", task)
        if model_checkpoint == "electra-base-discriminator":
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
        print("===========================================")
        print("Start test model --", model_checkpoint,  " on task--", task)

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


        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format('torch')
        train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=128)





        model.eval()
        predictions = []


        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)
        # Move the model to the device
        model = model.to(device)
        logits_list=[]
        responses= []

        for batch in tqdm(train_dataloader):
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
        print("Accuracy is :", responses_arr.mean())

        filename = f"{model_checkpoint}_{task}_{max_length}_response_logits_Accuracy_{responses_arr.mean()}.json"

        data = {
            "logits": logits_arr.tolist(),
            "responses": responses_arr.tolist()
        }

        with open(filename, "w") as f:
            json.dump(data, f)



