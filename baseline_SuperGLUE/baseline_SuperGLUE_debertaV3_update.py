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
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.optim import AdamW
from datasets import load_metric
from transformers import get_linear_schedule_with_warmup
import json
import evaluate
from torch.nn.utils.rnn import pad_sequence
from transformers import get_scheduler

# Set random seed for reproducibility
random_seed = 21
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_dir = "~/.cache/SuperGLUE"
models = ["deberta-v3-base"]
SUPERGLUE_TASKS = ["cb", "copa", "rte", "wic", "wsc", "boolq"]
GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available: ", GPU_avail)

# Directory to store results
output_dir = "superglue_baseline_debertaV3_base"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels  # changed from 'label' to 'labels'
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    if task == "cb":
        metric = evaluate.load("super_glue", "cb", trust_remote_code=True)
        results = metric.compute(predictions=predictions, references=labels)
        return {"accuracy": results["accuracy"]}
    elif task == "copa":
        metric = evaluate.load("super_glue", "copa", trust_remote_code=True)
        return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
    elif task == "rte":
        metric = evaluate.load("super_glue", "rte", trust_remote_code=True)
        return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
    elif task == "wic":
        metric = evaluate.load("super_glue", "wic", trust_remote_code=True)
        return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
    elif task == "wsc":
        metric = evaluate.load("super_glue", "wsc", trust_remote_code=True)
        return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}
    elif task == "boolq":
        metric = evaluate.load("super_glue", "boolq", trust_remote_code=True)
        return {"accuracy": metric.compute(predictions=predictions, references=labels)["accuracy"]}

train_batch_size = 24
dev_batch_size = 24
num_epochs = 20  # Increase the number of epochs to allow more training
max_length = 512
num_workers = 4  # Number of workers for data loading

for task in SUPERGLUE_TASKS:
    print(f"\nTask: {task}")
    dataset = load_dataset("super_glue", task, trust_remote_code=True)
    print(f"Dataset Keys: {dataset.keys()}")

    # Split the training set into 90% training and 10% validation
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=random_seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    test_dataset = dataset["validation"] if task != "multirc" else dataset["test"]

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    for model_checkpoint in models:
        print("===========================================")
        print("Start test model --", model_checkpoint, " on task --", task)
        best_model_dir = f"{output_dir}/best_model_{model_checkpoint}_{task}"
        # Check if the stored model and tokenizer exist
        if os.path.exists(best_model_dir):
            print(f"Loading the best model from {best_model_dir}")
            model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
            tokenizer = AutoTokenizer.from_pretrained(best_model_dir, use_fast=False)
        else:
            print(f"Initializing new model {model_checkpoint} for task {task}")
            tokenizer = AutoTokenizer.from_pretrained(f'microsoft/{model_checkpoint}', use_fast=False)
            num_labels = len(set(train_dataset['label'])) if 'label' in train_dataset.features else 1

            model = AutoModelForSequenceClassification.from_pretrained(f'microsoft/{model_checkpoint}',
                                                                       num_labels=num_labels)

        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        tokenized_train_dataset.set_format('torch')
        tokenized_val_dataset.set_format('torch')
        tokenized_test_dataset.set_format('torch')

        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=train_batch_size, shuffle=True,
                                      num_workers=num_workers, collate_fn=collate_fn)
        val_dataloader = DataLoader(tokenized_val_dataset, batch_size=dev_batch_size, shuffle=False,
                                    num_workers=num_workers, collate_fn=collate_fn)
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=dev_batch_size, shuffle=False,
                                     num_workers=num_workers, collate_fn=collate_fn)

        device = torch.device("cuda" if torch.cuda.is_available() else "situation")
        print(device)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

        # Define number of training steps
        num_training_steps = len(train_dataloader) * 8

        # Create a linear scheduler with warm-up
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_training_steps * 0.1,  # Warm-up for 10% of steps
            num_training_steps=num_training_steps
        )

        best_metric = 0.0
        early_stop_count = 0
        patience = 3  # Adjusted patience for early stopping

        training_stats = []
        detailed_training_stats = []

        time_s = time.time()
        for epoch in range(num_epochs):
            print(f"\n======== Epoch {epoch + 1} / {num_epochs} ========")
            print("Training...")
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(train_dataloader)):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)  # changed from 'label' to 'labels'

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Record detailed training loss every 10 steps
                if step % 10 == 0:
                    detailed_training_stats.append({
                        'epoch': epoch + 1,
                        'step': step,
                        'Training Loss': loss.item()
                    })

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            print("\nRunning Validation...")
            model.eval()
            val_loss = 0
            eval_predictions = []
            eval_labels = []

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            print("\nRunning Validation...")
            model.eval()
            val_loss = 0
            eval_predictions = []
            eval_labels = []

            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)  # changed from 'label' to 'labels'

                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()

                    logits = outputs.logits
                    eval_predictions.append(logits.cpu().numpy())
                    eval_labels.append(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_dataloader)
            eval_predictions = np.concatenate(eval_predictions, axis=0)
            eval_labels = np.concatenate(eval_labels, axis=0)

            eval_metrics = compute_metrics((eval_predictions, eval_labels))
            eval_accuracy = eval_metrics['accuracy']

            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {eval_accuracy:.4f}")

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Validation Loss': avg_val_loss,
                    'Validation Accuracy': eval_accuracy
                }
            )

            if eval_accuracy > best_metric:
                best_metric = eval_accuracy
                early_stop_count = 0
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print("Early stopping triggered")
                    break

        time_e = time.time()

        # Load the best model and tokenizer for testing
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        model.to(device)

        print("\nRunning Test (on dev set)...")
        model.eval()
        test_predictions = []
        test_labels = []

        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # changed from 'label' to 'labels'

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                test_predictions.append(logits.cpu().numpy())
                test_labels.append(labels.cpu().numpy())

        test_predictions = np.concatenate(test_predictions, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        test_metrics = compute_metrics((test_predictions, test_labels))
        test_accuracy = test_metrics['accuracy']

        print(f"Test Accuracy: {test_accuracy:.4f}")

        train_time = time_e - time_s
        actual_epochs = epoch + 1
        print(f"Total Training Time: {train_time:.2f} seconds")

        final_stats_filename = f"{output_dir}/final_stats_{model_checkpoint}_{task}_{train_time:.2f}s_{actual_epochs}epochs_{test_accuracy:.4f}.json"
        with open(final_stats_filename, "w") as f:
            json.dump(training_stats, f)
