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
from transformers import AutoTokenizer, T5ForSequenceClassification, TrainingArguments, Trainer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import evaluate

# Set random seed for reproducibility
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

models = ["t5-base"]
SUPERGLUE_TASKS = ["cb", "copa", "rte", "wic", "wsc", "boolq"]
GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available: ", GPU_avail)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to store results
output_dir = "superglue_baseline_t5_2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def tokenize_function(examples, tokenizer, task, max_length):
    if task == "cb":
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=max_length)
    elif task == "copa":
        return tokenizer(examples["premise"], examples["choice1"], examples["choice2"], padding="max_length", truncation=True, max_length=max_length)
    elif task == "rte":
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=max_length)
    elif task == "wic":
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=max_length)
    elif task == "wsc":
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    elif task == "boolq":
        return tokenizer(examples["question"], examples["passage"], padding="max_length", truncation=True, max_length=max_length)

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

train_batch_size = 36
dev_batch_size = 36
num_epochs = 30
max_length = 128
num_workers = 4

for task in SUPERGLUE_TASKS:
    print(f"\nTask: {task}")
    dataset = load_dataset("super_glue", task, trust_remote_code=True)
    print(f"Dataset Keys: {dataset.keys()}")

    # Split the training set into 90% training and 10% validation
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=random_seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    test_dataset = dataset["validation"]

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    for model_checkpoint in models:
        print("===========================================")
        print("Start test model --", model_checkpoint, " on task --", task)

        # Define the model and tokenizer paths
        best_model_dir = f"{output_dir}/best_model_{model_checkpoint}_{task}"

        # Check if the stored model and tokenizer exist
        if os.path.exists(best_model_dir):
            print(f"Loading the best model from {best_model_dir}")
            model = T5ForSequenceClassification.from_pretrained(best_model_dir)
            tokenizer = AutoTokenizer.from_pretrained(best_model_dir, model_max_length=max_length)
        else:
            print(f"Initializing new model {model_checkpoint} for task {task}")
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=max_length)
            num_labels = 3 if task == "cb" else 2
            model = T5ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

        tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer, task, max_length), batched=True)
        tokenized_val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer, task, max_length), batched=True)
        tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, task, max_length), batched=True)

        tokenized_train_dataset.set_format('torch')
        tokenized_val_dataset.set_format('torch')
        tokenized_test_dataset.set_format('torch')

        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(tokenized_val_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=num_workers)
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=dev_batch_size, shuffle=False, num_workers=num_workers)

        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.999))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * num_epochs)

        best_accuracy = 0.0
        early_stop_count = 0
        patience = 3  # Stop if performance doesn't improve for 3 consecutive epochs

        training_stats = []
        detailed_training_stats = []

        time_s = time.time()
        for epoch in range(num_epochs):
            print(f"\n======== Epoch {epoch + 1} / {num_epochs} ========")
            print("Training...")

            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                # Clear GPU cache
                torch.cuda.empty_cache()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

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
            eval_metric = evaluate.load("accuracy")
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    eval_metric.add_batch(predictions=predictions, references=labels)
                    val_loss += outputs.loss.item()

            val_loss = val_loss / len(val_dataloader)

            eval_score = eval_metric.compute()
            validation_accuracy = eval_score['accuracy']
            print(f"Validation Accuracy: {validation_accuracy:.4f}")

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Validation Loss': val_loss,
                    'Validation Accuracy': validation_accuracy
                }
            )

            # Save training stats at each epoch
            training_stats_filename = f"{output_dir}/training_stats_{model_checkpoint}_{task}.json"
            with open(training_stats_filename, "w") as f:
                json.dump(training_stats, f)

            # Save detailed training stats
            detailed_training_stats_filename = f"{output_dir}/detailed_training_stats_{model_checkpoint}_{task}.json"
            with open(detailed_training_stats_filename, "w") as f:
                json.dump(detailed_training_stats, f)

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                early_stop_count = 0
                # Save the best model and tokenizer
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print("Early stopping triggered")
                    break

        time_e = time.time()

        # Load the best model for testing
        model = T5ForSequenceClassification.from_pretrained(best_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        model.to(device)

        # Testing (on dev set)
        print("\nRunning Test (on dev set)...")
        model.eval()
        eval_metric = evaluate.load("accuracy")
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                eval_metric.add_batch(predictions=predictions, references=labels)

        eval_score = eval_metric.compute()
        test_accuracy = eval_score['accuracy']
        print(f"Test Accuracy: {test_accuracy:.4f}")

        train_time = time_e - time_s
        actual_epochs = epoch + 1
        print(f"Total Training Time: {train_time:.2f} seconds")

        # Save final training stats
        final_stats_filename = f"{output_dir}/final_stats_{model_checkpoint}_{task}_{train_time:.2f}s_{actual_epochs}epochs_{test_accuracy:.4f}.json"
        with open(final_stats_filename, "w") as f:
            json.dump(training_stats, f)
