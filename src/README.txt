This supplement includes code used for the AAAI 2020 submission "Dynamic Data Selection for Curriculum Learning via Ability Estimation"

Included in the supplement are the following files:

1. experiment files: cifar.py, mnist.py, snli.py, and sstb.py implement model training for each of the data sets considered in the submission. In particular, each file shows how model competency is estimated at the beginning of a training epoch, and how that estimate is used to select training data.

2. ability estimation: irt_scoring.py implements the ability estimation given a data set with known difficulties.

3. Data selection: build_features.py, specifically the get_epoch_training_data and get_epoch_training_data_vision methods, implement the data selection code given a task and an estimated ability level, as well as several baselines.

The provided code cannot be run as-is, because it requires a number of additional libraries for data processing and other tasks. A full, working implementation will be released as open-source software upon publication.
