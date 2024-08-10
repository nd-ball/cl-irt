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
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_scheduler
import json
import evaluate
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
import optuna

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

models = ["gpt2"]
SUPERGLUE_TASKS = ["cb", "copa", "rte", "wic", "wsc", "boolq"]
GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available: ", GPU_avail)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to store results
output_dir = "superglue_baseline_gpt2_1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

best_hyperparameters = {}

def tokenize_function(examples, tokenizer, task, max_length):
    if task == "cb":
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=max_length)
    elif task == "copa":
        return tokenizer(examples["premise"], examples["choice1"], examples["choice2"], truncation=True, padding="max_length", max_length=max_length)
    elif task == "rte":
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=max_length)
    elif task == "wic":
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=max_length)
    elif task == "wsc":
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
    elif task == "boolq":
        return tokenizer(examples["question"], examples["passage"], truncation=True, padding="max_length", max_length=max_length)

def collate_fn(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
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
        best_model_dir = f"{output_dir}/best_model_{model_checkpoint}_{task}"

        # Hyperparameter tuning with Optuna
        def objective(trial):
            lr = trial.suggest_categorical('lr', [1.5e-5, 2e-5, 2.5e-5, 3e-5])
            weight_decay = trial.suggest_float('weight_decay', 0.01, 0.01, log=True)
            dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.15])
            warmup_steps = trial.suggest_categorical('warmup_steps', [50, 100, 500, 1000])

            tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            num_labels = 3 if task == "cb" else 2
            model = GPT2ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
            model.config.hidden_dropout_prob = dropout
            model.config.attention_probs_dropout_prob = dropout
            model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match new tokenizer
            model = model.to(device)

            tokenized_train_dataset = train_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, task, max_length),
                batched=True
            )
            tokenized_val_dataset = val_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, task, max_length),
                batched=True
            )
            tokenized_test_dataset = test_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, task, max_length),
                batched=True
            )

            tokenized_train_dataset.set_format('torch')
            tokenized_val_dataset.set_format('torch')

            train_dataloader = DataLoader(tokenized_train_dataset, batch_size=24, shuffle=True,
                                          num_workers=num_workers,
                                          collate_fn=lambda batch: collate_fn(batch, tokenizer))
            val_dataloader = DataLoader(tokenized_val_dataset, batch_size=24, shuffle=False,
                                        num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, tokenizer))

            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            num_training_steps = len(train_dataloader) * 20
            scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )

            scaler = GradScaler()

            for epoch in range(20):
                model.train()
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    with autocast():
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

            model.eval()
            eval_predictions = []
            eval_labels = []
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    eval_predictions.extend(logits.cpu().numpy())
                    eval_labels.extend(labels.cpu().numpy())

            eval_predictions = np.array(eval_predictions)
            eval_labels = np.array(eval_labels)
            eval_results = compute_metrics((eval_predictions, eval_labels))
            return eval_results["accuracy"]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)  # Adjust the number of trials as needed

        # Use the best hyperparameters found by Optuna
        best_params = study.best_params
        lr = best_params['lr']
        weight_decay = best_params['weight_decay']
        dropout = best_params['dropout']
        warmup_steps = best_params['warmup_steps']

        print(f"Best hyperparameters: {best_params}")

        # Store the best hyperparameters for the current task
        best_hyperparameters[task] = best_params

        # Initialize model and tokenizer with the best hyperparameters
        tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        num_labels = 3 if task == "cb" else 2
        model = GPT2ForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
        model.config.hidden_dropout_prob = dropout
        model.config.attention_probs_dropout_prob = dropout
        model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match new tokenizer
        model = model.to(device)

        tokenized_train_dataset = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, task, max_length),
            batched=True
        )
        tokenized_val_dataset = val_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, task, max_length),
            batched=True
        )
        tokenized_test_dataset = test_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer, task, max_length),
            batched=True
        )

        tokenized_train_dataset.set_format('torch')
        tokenized_val_dataset.set_format('torch')
        tokenized_test_dataset.set_format('torch')

        train_dataloader = DataLoader(tokenized_train_dataset, batch_size=24, shuffle=True,
                                      num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, tokenizer))
        val_dataloader = DataLoader(tokenized_val_dataset, batch_size=24, shuffle=False,
                                    num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, tokenizer))
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=24, shuffle=False,
                                     num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch, tokenizer))

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Define number of training steps
        num_training_steps = len(train_dataloader) * 20

        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()

        best_metric = 0.0
        early_stop_count = 0
        patience = 3  # Adjusted patience for early stopping

        training_stats = []
        detailed_training_stats = []

        time_s = time.time()
        for epoch in range(20):
            print(f"\n======== Epoch {epoch + 1} / 20 ========")
            print("Training...")
            model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                # Mixed precision training
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()

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

            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    eval_predictions.extend(logits.cpu().numpy())
                    eval_labels.extend(labels.cpu().numpy())
                    val_loss += outputs.loss.item()

            val_loss = val_loss / len(val_dataloader)
            eval_predictions = np.array(eval_predictions)
            eval_labels = np.array(eval_labels)

            eval_results = compute_metrics((eval_predictions, eval_labels))
            print(f"Validation Metrics: {eval_results}")

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Validation Loss': val_loss,
                    'Validation Accuracy': eval_results["accuracy"]
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

            current_metric = eval_results["accuracy"]
            if current_metric > best_metric:
                best_metric = current_metric
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
        model = GPT2ForSequenceClassification.from_pretrained(best_model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(best_model_dir)
        model.to(device)

        # Testing (on dev set)
        print("\nRunning Test (on dev set)...")
        model.eval()
        test_predictions = []
        test_labels = []

        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                test_predictions.extend(logits.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)

        test_results = compute_metrics((test_predictions, test_labels))
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")

        train_time = time_e - time_s
        actual_epochs = epoch + 1
        print(f"Total Training Time: {train_time:.2f} seconds")

        # Save final training stats
        final_stats_filename = f"{output_dir}/final_stats_{model_checkpoint}_{task}_{train_time:.2f}s_{actual_epochs}epochs_{test_results['accuracy']:.4f}.json"
        with open(final_stats_filename, "w") as f:
            json.dump(training_stats, f)

# Save the best hyperparameters for all tasks to a JSON file
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hyperparameters, f)
