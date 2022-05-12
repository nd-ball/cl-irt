"""
Workhorse file for loading everything, setting parameters and training
all experiments will be defined via a TOML file
"""

import typer
import toml
from rich.console import Console

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset, load_metric
    
from models.models import MODELS
#from datasets.datasets import DATASETS 
from trainers.trainers import TRAINERS
from teachers.default_teacher import DefaultTeacher

console = Console()
app = typer.Typer()


@app.command()
def train(
    config_path: str
):
    with open(config_path, "r") as f:
        experiment_config = toml.load(f)

    # load data set
    dataset = load_dataset(experiment_config["data"]["name"])

    # assign ids (always do this for consistency)
    dataset = dataset.map(lambda x, idx: {"idx": idx}, with_indices=True) 

    # load initial difficulties
    difficulties = load_difficulties(experiment_config["data"]["diff_file"])
    dataset = dataset.map(lambda x: {"difficulty": difficulties[x["idx"]]}) 
    
    # load and initialize model
    model = MODELS[
        experiment_config["model"]["name"]
    ](experiment_config)

    # load and initialize trainer 
    trainer = TRAINERS[
        experiment_config["trainer"]["name"]
    ](experiment_config)

    # load the teacher 
    teacher = DefaultTeacher(model, data, trainer, experiment_config)

    # train model 
    teacher.train() 

@app.command()
def test(
    config_path: str
):
    pass

@app.command()
def run_crowd(
    config_path: str
):
    with open(config_path, "r") as f:
        experiment_config = toml.load(f)

    # load data set and assign ids
    dataname = experiment_config["data"]["name"]
    dataset = load_dataset(dataname)
    dataset = dataset.map(lambda x, idx: {"idx": idx}, with_indices=True)

    # format dataset
    model_name = experiment_config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_dataset["train"].shuffle(seed=42)
    eval_dataset = tokenized_dataset["test"].shuffle(seed=42)

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    crowd_size = experiment_config["model"]["crowd_size"]
    for i in range(crowd_size):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(set(train_dataset["label"]))
        )
        
        training_args = TrainingArguments(
            output_dir=f"{dataname}_{i}",
            evaluation_strategy="epoch"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()


if __name__ == "__main__":
    app()
