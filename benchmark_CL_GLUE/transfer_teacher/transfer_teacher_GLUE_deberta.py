import os
import warnings

# Suppress TensorFlow warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging to errors only
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

import numpy as np
import random
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import evaluate
import time
import torch.optim as optim

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.convert_slow_tokenizer')

# Set random seed for reproducibility
random_seed = 63
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_checkpoint = "deberta-v3-base"
GLUE_TASKS = ["mrpc", "rte", "mnli", "qqp", "sst2", "qnli"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def tokenize_function(examples, task, tokenizer, max_length):
    if task in ["mnli"]:
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=max_length)
    elif task in ["mrpc", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=max_length)
    elif task in ["qnli"]:
        return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True, max_length=max_length)
    elif task in ["qqp"]:
        return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True, max_length=max_length)
    elif task in ["sst2"]:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

def calculate_qap(model, dataloader, device):
    model.to(device)
    model.eval()
    qap_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating QAP"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

            correct_probs = probabilities[torch.arange(probabilities.size(0), device=device), labels]
            qap_scores.extend(correct_probs.cpu().numpy())

    return np.array(qap_scores)

def sort_dataset_by_qap(dataset, qap_scores):
    sorted_indices = np.argsort(qap_scores)
    sorted_dataset = dataset.select(sorted_indices)
    return sorted_dataset.add_column("qap", qap_scores[sorted_indices])

def evaluate_model(model, dataloader, device):
    model.eval()
    eval_metric = evaluate.load("accuracy")
    for batch in tqdm(dataloader):
        # Debugging: print the keys of the batch
        # print(batch.keys())
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            eval_metric.add_batch(predictions=predictions, references=labels)
    return eval_metric.compute()["accuracy"]

def train_and_evaluate(task, model, tokenizer, train_dataset, val_dataset, test_dataset, starting_percent, inc, step_length, alpha, device, output_dir):
    model.to(device)

    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 10
    max_steps = 10000
    logging_steps = 500

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = min(max_steps, int(len(train_dataset) * num_epochs / train_batch_size))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    global_step = 0
    curr_percent = starting_percent
    best_accuracy = 0.0
    start_time = time.time()

    training_stats = []
    detailed_training_stats = []

    patience = 3  # Set patience for early stopping
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        curr_idxs = list(range(0, int(curr_percent * len(train_dataset))))
        curr_dataloader = DataLoader(Subset(train_dataset, curr_idxs), batch_size=train_batch_size, shuffle=False)

        for step, batch in enumerate(tqdm(curr_dataloader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            detailed_training_stats.append(
                {
                    'step': global_step,
                    'loss': loss.item()
                }
            )

            if global_step % step_length == 0:
                if curr_percent < 1:
                    curr_percent = min(starting_percent * (inc ** (global_step / step_length)), 1)
                    curr_idxs = list(range(0, int(curr_percent * len(train_dataset))))
                    curr_dataloader = DataLoader(Subset(train_dataset, curr_idxs), batch_size=train_batch_size, shuffle=False)

                    print(f"At step {global_step}, usage percent changed to {curr_percent}")
                    print(f"Current dataloader length: {len(curr_dataloader)}")

            if global_step % logging_steps == 0:
                print(f"Step {global_step}: Loss: {loss.item():.4f}")

            if global_step >= max_steps:
                break

        # Evaluate at the end of each epoch
        val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
        val_accuracy = evaluate_model(model, val_dataloader, device)
        avg_train_loss = total_loss / len(curr_dataloader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        training_stats.append(
            {
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Validation Accuracy': val_accuracy,
                'Epoch Time': epoch_duration
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

        best_model_dir = f"{output_dir}/best_model_{model_checkpoint}_{task}"
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            early_stop_count = 0
            # Save the best model and tokenizer
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered")
                break

        if global_step >= max_steps:
            break

    # Load the best model for final evaluation
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model.to(device)

    # Final evaluation on test set
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    test_accuracy = evaluate_model(model, test_dataloader, device)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    # Save final training stats
    final_stats_filename = f"{output_dir}/final_stats_{model_checkpoint}_{task}_{training_time:.2f}s_{num_epochs}epochs_{test_accuracy:.4f}.json"
    with open(final_stats_filename, "w") as f:
        json.dump(training_stats, f)

    return test_accuracy, training_time

if __name__ == "__main__":
    # Set random seed for reproducibility
    random_seed = 21
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_checkpoint = "deberta-v3-base"
    GLUE_TASKS = ["mrpc", "rte", "mnli", "qqp", "sst2", "qnli"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory to store results
    output_dir = "glue_transfer_teacher_debertaV3_2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Main execution
    for task in GLUE_TASKS:
        print(f"\nTask: {task}")
        dataset = load_dataset("glue", task)

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

        best_model_dir = f"./teacher_best_model/best_model_{model_checkpoint}_{task}"

        # Load teacher model or initialize student model
        if os.path.exists(best_model_dir):
            print(f"Loading the best teacher model from {best_model_dir}")
            teacher_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
            teacher_model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        else:
            print(f"Initializing new student model {model_checkpoint} for task {task}")
            tokenizer = AutoTokenizer.from_pretrained(f'microsoft/{model_checkpoint}')
            num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
            teacher_model = AutoModelForSequenceClassification.from_pretrained(f'microsoft/{model_checkpoint}', num_labels=num_labels)
            teacher_model.to(device)

        # Ensure datasets are tokenized and formatted
        train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, 128), batched=True)
        val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, 128), batched=True)
        test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, 128), batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # Print accuracy of the teacher model on the training dataset
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        teacher_train_accuracy = evaluate_model(teacher_model, train_dataloader, device)
        print(f"Teacher model training accuracy: {teacher_train_accuracy:.4f}")

        tokenizer = AutoTokenizer.from_pretrained(f'microsoft/{model_checkpoint}')
        num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
        model = AutoModelForSequenceClassification.from_pretrained(f'microsoft/{model_checkpoint}', num_labels=num_labels)
        model.to(device)

        # Calculate initial QAP scores and sort the dataset
        qap_scores = calculate_qap(teacher_model, train_dataloader, device)
        sorted_train_dataset = sort_dataset_by_qap(train_dataset, qap_scores)

        # Train and evaluate without Bayesian Optimization
        starting_percent = 0.26
        inc = 1.5
        step_length = 2
        alpha = 0.5
        test_accuracy, training_time = train_and_evaluate(task, model, tokenizer, sorted_train_dataset, val_dataset, test_dataset, starting_percent, inc, step_length, alpha, device, output_dir)

        # print(f"Best ACL parameters for {task}:")
        print(f"Test Accuracy: {test_accuracy:.4f}, Training Time: {training_time:.2f} seconds")

    print("All tasks completed.")
