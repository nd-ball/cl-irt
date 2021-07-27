"""
Workhorse file for loading everything, setting parameters and training
all experiments will be defined via a TOML file
"""

import typer
import toml
from rich.console import Console

from models.models import MODELS
from datasets.datasets import DATASETS 
from trainers.trainers import TRAINERS

console = Console()
app = typer.Typer()


@app.command
def train(
    config_path: str
):
    with open(config_path, "r") as f:
        experiment_config = toml.load(f)

    # load data set
    data = DATASETS[
        experiment_config.data["name"]
    ](experiment_config)

    # load and initialize model
    model = MODELS[
        experiment_config.model["name"]
    ](experiment_config)

    # load and initialize trainer 
    trainer = TRAINERS[
        experiment_config.trainer["name"]
    ](model, data, experiment_config)

    # train model 
    trainer.train() 

