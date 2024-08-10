import numpy as np
import argparse
import random
import os
import time
import json
import torch
import transformers
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoModelForSequenceClassification, \
    AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from datasets import load_dataset, Dataset
import evaluate
from build_features import get_epoch_training_data, get_example_rarities
from irt_scoring import calculate_theta

# Set batch size and max sequence length at the beginning
batch_size = 128
max_seq_len = 128

transformers.logging.set_verbosity_error()

def load_and_prepare_data(task, diff_dir):
    raw_datasets = load_dataset('glue', task)

    train_diff_file = f'{diff_dir}/{task.lower()}1pl/best_parameters.json'
    with open(train_diff_file, 'r') as file:
        data = json.load(file)

    train = raw_datasets['train']
    train = train.add_column('difficulty', data['diff'])


    random_seed = 21
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    train_val_split = train.train_test_split(test_size=0.1, seed=random_seed)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    test_dataset = raw_datasets['validation_matched'] if task == 'mnli' else raw_datasets['validation']

    return train_dataset, val_dataset, test_dataset

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
    else:
        raise ValueError(f"Task {task} not supported for tokenization")


def create_dataset(dataset, task, tokenizer, max_length, include_difficulty=True):
    # Map the tokenize function to the dataset
    tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, task, tokenizer, max_length), batched=True)

    # Set the format and return the TensorDataset
    if include_difficulty:
        tokenized_dataset.set_format(type='torch',
                                     columns=['input_ids', 'attention_mask',
                                              'token_type_ids', 'label', 'difficulty'])

        return TensorDataset(
            tokenized_dataset['input_ids'],
            tokenized_dataset['attention_mask'],
            tokenized_dataset['token_type_ids'],
            tokenized_dataset['label'],
            tokenized_dataset['difficulty']
        )
    else:
        tokenized_dataset.set_format(type='torch',
                                     columns=['input_ids', 'attention_mask',
                                              'token_type_ids', 'label'])

        return TensorDataset(
            tokenized_dataset['input_ids'],
            tokenized_dataset['attention_mask'],
            tokenized_dataset['token_type_ids'],
            tokenized_dataset['label']
        )

def evaluate_and_estimate(model, dataloader, device, mode='eval'):
    val_loss = 0
    eval_metric = evaluate.load("accuracy")
    preds, out_label_ids = None, None

    model.eval()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                labels=batch[3]
            )
            logits = outputs.logits

        preds = np.append(preds, logits.detach().cpu().numpy(),
                          axis=0) if preds is not None else logits.detach().cpu().numpy()
        out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(),
                                  axis=0) if out_label_ids is not None else batch[3].detach().cpu().numpy()

        predictions = torch.argmax(logits, dim=-1)
        eval_metric.add_batch(predictions=predictions, references=batch[3])
        val_loss += outputs.loss.item()



    val_loss /= len(dataloader)
    eval_score = eval_metric.compute()
    validation_accuracy = eval_score['accuracy']

    if mode == 'eval':
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        return validation_accuracy, val_loss
    elif mode == 'estimate':
        time_model_s = time.time()  # Start timing here
        rps = [1 if p == c else -1 for p, c in zip(np.argmax(preds, axis=1), out_label_ids)]
        theta_hat = calculate_theta(dataloader.dataset.tensors[4].cpu().numpy(), rps)[0]
        time_model_e = time.time()  # End timing here
        model_capacity_time = time_model_e - time_model_s
        return theta_hat, model_capacity_time
    elif mode == 'eval_estimate':
        time_model_s = time.time()  # Start timing here
        rps = [1 if p == c else -1 for p, c in zip(np.argmax(preds, axis=1), out_label_ids)]
        theta_hat = calculate_theta(dataloader.dataset.tensors[4].cpu().numpy(), rps)[0]
        time_model_e = time.time()  # End timing here
        model_capacity_time = time_model_e - time_model_s
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        return validation_accuracy, val_loss, theta_hat, model_capacity_time


def train(args, output_dir):
    print(f"\nTask: {args.task}")
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    train_dataset, dev_dataset, test_dataset = load_and_prepare_data(args.task, args.diff_dir)

    train_size = len(train_dataset)
    val_size = len(dev_dataset)
    test_size = len(test_dataset)
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Determine the number of labels dynamically
    num_labels = 3 if args.task.startswith("mnli") else 1 if args.task == "stsb" else 2

    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base',
                                                   cache_dir=args.cache_dir,
                                                   use_fast=False)
    model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=num_labels,
                                                               cache_dir=args.cache_dir)
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_stats = []
    detailed_training_stats = []

    train_dataset = create_dataset(train_dataset, args.task, tokenizer, max_seq_len, include_difficulty=True)
    dev_dataset = create_dataset(dev_dataset, args.task, tokenizer, max_seq_len, include_difficulty=True)
    test_dataset = create_dataset(test_dataset, args.task, tokenizer, max_seq_len, include_difficulty=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.999))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                num_training_steps=len(train_dataloader) * 5)

    best_accuracy = 0.0
    early_stop_count = 0
    patience = 3
    best_model_dir = os.path.join(output_dir, "best_model")

    time_s = time.time()
    total_model_capacity_time = 0.0  # Accumulate the total model capacity evaluation time

    prev_cap = -5
    cur_cap = 0

    for epoch in range(args.num_epochs):
        print(f"\n======== Epoch {epoch + 1} / {args.num_epochs} ========")
        print("Filtering the train dataset...")
        epoch_loss = 0.0

        if epoch == 0 and args.strategy == 'theta':
            theta_hat, model_capacity_time = evaluate_and_estimate(model, dev_dataloader, device, mode='estimate')
            total_model_capacity_time += model_capacity_time

            if theta_hat> prev_cap:
                cur_cap=theta_hat
                prev_cap=theta_hat
            else:
                print("Model didn't imporve")
                cur_cap +=0.1



        time_model_s = time.time()
        filtered_training_data = get_epoch_training_data(train_dataset, args, epoch, 'glue', cur_cap,
                                                         lower_offset=args.lower_bound, upper_offset=args.upper_bound)
        time_model_e = time.time()
        model_capacity_time += time_model_e - time_model_s  # Add time for getting epoch training data
        total_model_capacity_time += time_model_e - time_model_s



        l = len(filtered_training_data['labels'])
        print(f"\n Current model capacity is {cur_cap} and {l} training data have been selected ...")

        # Directly create the TensorDataset from filtered_training_data
        train_dataset_epoch = TensorDataset(
            filtered_training_data['input_ids'],
            filtered_training_data['attention_mask'],
            filtered_training_data['token_type_ids'],
            filtered_training_data['labels'],
            filtered_training_data['difficulty']
        )
        if args.task in ['rte','mrpc']:
            train_dataloader_epoch = DataLoader(train_dataset_epoch, shuffle=True,
                                                batch_size=batch_size, num_workers=args.num_workers)
        else:
            train_dataloader_epoch = DataLoader(train_dataset_epoch,
                                                batch_size=batch_size, num_workers=args.num_workers)
        print("Training...")
        model.train()
        for step, batch in enumerate(train_dataloader_epoch):
            batch = tuple(t.to(device) for t in batch)
            model.zero_grad()
            # Clear GPU cache
            # torch.cuda.empty_cache()
            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                labels=batch[3]
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % 10 == 0:
                detailed_training_stats.append({
                    'epoch': epoch + 1,
                    'step': step,
                    'Training Loss': loss.item()
                })

        avg_train_loss = epoch_loss / len(train_dataloader_epoch)

        dev_acc, val_loss, theta_hat, model_capacity_time = evaluate_and_estimate(model, dev_dataloader, device,
                                                                                  mode='eval_estimate')
        total_model_capacity_time += model_capacity_time
        # Accumulate the time
        if theta_hat > prev_cap:
            cur_cap = theta_hat
            prev_cap = theta_hat
        else:
            print("Model didn't imporve")
            cur_cap += 0.1

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': val_loss,
            'Validation Accuracy': dev_acc,
            'Model Capacity Evaluation Time': model_capacity_time,
            'Number of Training Examples': l
        })

        if dev_acc > best_accuracy:
            best_accuracy = dev_acc
            early_stop_count = 0
            # Save the best model and tokenizer
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)  # Optionally save the tokenizer for consistency
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered")
                break

    time_e = time.time()
    train_time = time_e - time_s
    actual_epochs = epoch + 1

    print(f"Total Training Time: {train_time:.2f} seconds")

    model_checkpoint = "deberta-v3-base"
    task = args.task

    training_stats_filename = f"{output_dir}/training_stats_{model_checkpoint}_{task}.json"
    with open(training_stats_filename, "w") as f:
        json.dump(training_stats, f)

    detailed_training_stats_filename = f"{output_dir}/detailed_training_stats_{model_checkpoint}_{task}.json"
    with open(detailed_training_stats_filename, "w") as f:
        json.dump(detailed_training_stats, f)

    # Load the best model and tokenizer for testing (reloading the tokenizer is optional if consistent environment)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir,use_fast=False)  # Optional
    model.to(device)

    test_acc, _ = evaluate_and_estimate(model, test_dataloader, device, mode='eval')
    print(f'Test accuracy after early stopping: {test_acc}')

    # Save final training stats
    final_stats_filename = f"{output_dir}/final_stats_{model_checkpoint}_{task}_Training:{train_time:.2f}s_PUDF:{total_model_capacity_time:.2f}s_{actual_epochs}epochs_{test_acc:.4f}.json"
    with open(final_stats_filename, "w") as f:
        json.dump(training_stats, f)

    return best_accuracy, test_acc


def run():
    GLUETASKS = ['mrpc','rte','sst2',  'mnli', 'qnli',   'qqp']
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='use GPU?')
    parser.add_argument('--data-dir', help='path to GLUE dataset')
    parser.add_argument('--diff-dir', help='path to difficulty dataset')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--strategy',
                        choices=['baseline', 'ordered', 'simple', 'theta', 'naacl-linear', 'naacl-root', 'theta-hard'],
                        default='simple')
    parser.add_argument('--ordering', choices=['easiest', 'hardest', 'middleout'], default='easiest')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--use-length', action='store_true')
    parser.add_argument('--use-word-rarity', action='store_true')
    parser.add_argument('--min-train-length', type=int, default=100)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--competency', type=int, default=50)
    parser.add_argument('--p-correct', type=float, default=0.5)
    parser.add_argument('--cache-dir', help='cache dir for Deberta models')
    parser.add_argument('--num-obs', type=int, default=1000)
    parser.add_argument('--task', choices=GLUETASKS, help='GLUE task for fine-tuning')
    parser.add_argument('--lower-bound', type=float, default=np.NINF)
    parser.add_argument('--upper-bound', type=float, default=0)
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers for data loading')
    args = parser.parse_args()

    output_dir = "./glue_PUDF_deberta_1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for task in GLUETASKS:
        args.task = task
        print(f"Starting training for task: {task}")
        start_time = time.time()
        top_dev, test_acc = train(args, output_dir)
        end_time = time.time()
        print(f'Total time for {task}: {end_time - start_time} seconds')
        print(f'Test accuracy for {task}: {test_acc}')


if __name__ == '__main__':
    run()
