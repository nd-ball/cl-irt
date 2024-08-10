import numpy as np
import pandas as pd
from datasets import load_dataset

SUPERGLUE_TASKS = ["cb", "copa", "rte", "wic", "wsc", "boolq"]


def check_and_map_labels(dataset, task):
    labels = dataset['label']
    unique_labels = set(labels)

    print(f"Task: {task}, Unique labels: {unique_labels}, Label counts: {pd.Series(labels).value_counts()}")

    if not all(isinstance(label, int) for label in unique_labels):
        # Map string labels to integers if necessary
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        print(f"Label mapping: {label_mapping}")
        dataset = dataset.map(lambda x: {'label': label_mapping[x['label']]})

    return dataset


def print_sample_examples(dataset, num_samples=3):
    print("Sample examples from the dataset:")
    for i in range(min(num_samples, len(dataset))):
        print(f"Example {i}:")
        print(f"  Input: {dataset[i]}")
        print(f"  Label: {dataset[i]['label']}")
    print("\n")


for task in SUPERGLUE_TASKS:
    print(f"\nTask: {task}")
    dataset = load_dataset("super_glue", task, trust_remote_code=True)

    print("Validation Set:")
    train_dataset = check_and_map_labels(dataset["validation"], task)
    print_sample_examples(train_dataset)

    print("Train Set:")
    pred_dataset = check_and_map_labels(dataset["train"], task)
    print_sample_examples(pred_dataset)
