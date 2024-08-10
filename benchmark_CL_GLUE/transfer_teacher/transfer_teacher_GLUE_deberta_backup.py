import numpy as np
import random
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from bayes_opt import BayesianOptimization
import evaluate
import time
import torch.optim as optim
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.convert_slow_tokenizer')

# Set random seed for reproducibility
random_seed = 21
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


def train_and_evaluate(task, model, tokenizer, train_dataset, val_dataset, test_dataset, starting_percent, inc,
                       step_length, alpha, device):
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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

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

            if global_step % step_length == 0:
                if curr_percent < 1:
                    print("Recomputing rank using adaptive CL")
                    curr_qap = calculate_qap(model, DataLoader(train_dataset, batch_size=eval_batch_size), device)
                    prev_qap = train_dataset["qap"]
                    new_qap = [(1 - alpha) * prev_qap[i] + (alpha * curr_qap[i]) for i in range(len(train_dataset))]

                    sort_index = np.argsort(new_qap)
                    train_dataset = train_dataset.select(sort_index)

                    # Remove existing qap column before adding the updated one
                    if "qap" in train_dataset.column_names:
                        train_dataset = train_dataset.remove_columns("qap")
                    train_dataset = train_dataset.add_column("qap", [float(new_qap[i]) for i in sort_index])

                    curr_percent = min(starting_percent * (inc ** (global_step / step_length)), 1)
                    curr_idxs = list(range(0, int(curr_percent * len(train_dataset))))
                    curr_dataloader = DataLoader(Subset(train_dataset, curr_idxs), batch_size=train_batch_size,
                                                 shuffle=False)

                    print(f"At step {global_step}, usage percent changed to {curr_percent}")
                    print(f"Current dataloader length: {len(curr_dataloader)}")

            if global_step % logging_steps == 0:
                print(f"Step {global_step}: Loss: {loss.item():.4f}")

            if global_step >= max_steps:
                break

        # Evaluate at the end of each epoch
        val_accuracy = evaluate_model(model, val_dataset, device)
        print(f"Epoch {epoch + 1} - Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Save the best model
            torch.save(model.state_dict(), f"best_model_{task}.pth")

        if global_step >= max_steps:
            break

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(f"best_model_{task}.pth"))

    # Final evaluation on test set
    test_accuracy = evaluate_model(model, test_dataset, device)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    return test_accuracy, training_time


def evaluate_model(model, dataset, device):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    metric = evaluate.load("accuracy")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)

    return metric.compute()["accuracy"]


def optimize_acl_params(task, model, tokenizer, train_dataset, val_dataset, test_dataset, device):
    def acl_wrapper(starting_percent, inc, step_length, alpha):
        starting_percent = round(starting_percent, 2)
        inc = round(inc, 2)
        step_length = 125 + int(250 * step_length)
        alpha = round(alpha, 2)

        accuracy, training_time = train_and_evaluate(task, model, tokenizer, train_dataset, val_dataset, test_dataset,
                                                     starting_percent, inc, step_length, alpha, device)
        # Save results
        results = {
            "task": task,
            "model": model_checkpoint,
            "best_accuracy": accuracy,
            "best_params": {
                "starting_percent": starting_percent,
                "inc": inc,
                "step_length": step_length,
                "alpha": alpha
            },
            "training_time": training_time
        }
        with open(f"{output_dir}/results_{task}_{model_checkpoint}.json", "w") as f:
            json.dump(results, f)
        return accuracy

    optimizer = BayesianOptimization(
        f=acl_wrapper,
        pbounds={
            "starting_percent": (0.1, 0.8),  # Start with more data
            "inc": (1.5, 3.0),  # Increase more aggressively
            "step_length": (0.01, 2),  # Slightly narrower range for step length
            "alpha": (0.5, 1)  # Higher alpha values
        },
        random_state=1234,
        verbose=2
    )

    optimizer.maximize(init_points=3, n_iter=15)
    return optimizer.max


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
    output_dir = "glue_transfer_teacher_debertaV3"
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
            teacher_model = AutoModelForSequenceClassification.from_pretrained(f'microsoft/{model_checkpoint}',
                                                                               num_labels=num_labels)
            teacher_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(f'microsoft/{model_checkpoint}')
        num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
        model = AutoModelForSequenceClassification.from_pretrained(f'microsoft/{model_checkpoint}',
                                                                   num_labels=num_labels)
        model.to(device)

        # Tokenize datasets
        train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, 128), batched=True)
        val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, 128), batched=True)
        test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, 128), batched=True)

        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        # Calculate initial QAP scores and sort the dataset
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        qap_scores = calculate_qap(teacher_model, train_dataloader, device)
        sorted_train_dataset = sort_dataset_by_qap(train_dataset, qap_scores)

        # Optimize ACL parameters
        best_params = optimize_acl_params(task, model, tokenizer, sorted_train_dataset, val_dataset, test_dataset, device)
        print(f"Best ACL parameters for {task}:")
        print(best_params)

    print("All tasks completed.")
