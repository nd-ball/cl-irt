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
from transformers import get_linear_schedule_with_warmup
import json
import evaluate

# Set random seed for reproducibility
random_seed = 21
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

models = ["deberta-v3-base"]
GLUE_TASKS = ["mrpc", "rte", "mnli", "qqp", "sst2", "qnli"]
GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available: ", GPU_avail)

# Directory to store results
output_dir = "glue_spl_deberta"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def tokenize_function(examples):
    if task in ["mnli"]:
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task in ["mrpc", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task in ["qnli"]:
        return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task in ["qqp"]:
        return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True,
                         max_length=max_length)
    elif task in ["sst2"]:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)


train_batch_size = 128
dev_batch_size = 128
num_epochs = 20  # Increase the number of epochs to allow more training
max_length = 128
num_workers = 4  # Number of workers for data loading


def calculate_confidence(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence = 1 - torch.max(probabilities, dim=-1)[0]  # Use 1 - max_prob as uncertainty
    return confidence


for task in GLUE_TASKS:
    print(f"\nTask: {task}")
    dataset = load_dataset("glue", task)
    print(f"Dataset Keys: {dataset.keys()}")

    # Split the training set into 90% training and 10% validation
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=random_seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    test_dataset = dataset["validation_matched"] if task == "mnli" else dataset["validation"]

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
            tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        else:
            print(f"Initializing new model {model_checkpoint} for task {task}")
            tokenizer = AutoTokenizer.from_pretrained(f'microsoft/{model_checkpoint}')
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            model = AutoModelForSequenceClassification.from_pretrained(f'microsoft/{model_checkpoint}',
                                                                       num_labels=num_labels)

        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

        tokenized_train_dataset.set_format('torch')
        tokenized_val_dataset.set_format('torch')
        tokenized_test_dataset.set_format('torch')

        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=train_batch_size, shuffle=True,
                                      num_workers=num_workers)
        val_dataloader = DataLoader(tokenized_val_dataset, batch_size=dev_batch_size, shuffle=False,
                                    num_workers=num_workers)
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=dev_batch_size, shuffle=False,
                                     num_workers=num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.999))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                    num_training_steps=len(train_dataloader) * num_epochs)

        best_accuracy = 0.0
        early_stop_count = 0
        patience = 5  # Adjusted patience for early stopping

        training_stats = []
        detailed_training_stats = []

        time_s = time.time()
        for epoch in range(num_epochs):
            print(f"\n======== Epoch {epoch + 1} / {num_epochs} ========")
            print("Training...")
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            # Calculate confidence scores for all samples
            all_confidences = []
            for batch in tqdm(train_dataloader, desc="Calculating confidences"):
                confidences = calculate_confidence(model, batch, device)
                all_confidences.extend(confidences.cpu().numpy())

            # Sort samples by confidence
            sorted_indices = np.argsort(all_confidences)

            # Determine the number of samples to use in this epoch
            num_samples = int((epoch + 1) / num_epochs * len(sorted_indices))
            selected_indices = sorted_indices[:num_samples]

            for step, batch in enumerate(tqdm(train_dataloader)):
                # Only use selected samples
                if step not in selected_indices:
                    continue

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Clear GPU cache
                torch.cuda.empty_cache()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Weight the loss by confidence
                confidence = calculate_confidence(model, batch, device)
                weighted_loss = loss * confidence.mean()

                total_loss += weighted_loss.item()

                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                # Record detailed training loss every 10 steps
                if step % 10 == 0:
                    detailed_training_stats.append({
                        'epoch': epoch + 1,
                        'step': step,
                        'Training Loss': weighted_loss.item()
                    })

            avg_train_loss = total_loss / num_samples
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
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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

        # Load the best model and tokenizer for testing
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
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
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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
