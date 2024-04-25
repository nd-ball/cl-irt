cl-irt
==============================

irt curriculum learning experiments

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


## 04/2024 Updates

High-level target: Write a pipeline code to calculate the difficulty on each dataset of GLUE or other datasets through pre-trained models with 0/1/3/5 epochs fine-tuning.

### There are three parts in the pipeline:
1. Generate accuracy
   
    Inputs: Datasets (GLUE), pre-trained models (e.g., albert, bert, t5, gpt-2) with 0/1/3/5 epochs fine-tuning

    Outputs: predicted accuracy, probability

2. Calculate difficulty of each dataset and ability of each model
   
    Inputs: accuracy

    Outputs: dataset difficulty, models's ability (through irt-model) 

3. Visulize results
   
    A. Difficulty distribution

    B. The probability of model labeling data correctly vs. difficulty

    C. Accuracy vs. probability 

    D. Predicted logits vs. difficulty

### Code
1. Generate accuracy
    
   Run `gen_respon_256.sh`

2. Calculate difficulty of each dataset and ability of each model

    Once obtained the predicted results, we do the following steps to generate the difficulty and ability:
   
    A. Run `cal_diff.py`: Merge the accuracy results of different models for each dataset.
   
    B. Run `generate_diff.py`: Convert the csv file into json file.
   
    C. Run `GLUE_pyirt.ipynb`: Generate data difficulty and model ability through py-irt library.

3. Visulize results: All the plot_xxx.py files



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
