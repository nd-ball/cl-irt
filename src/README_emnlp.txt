This zip file includes the code needed to run the experiments reported in this EMNLP submission.

1. experiment files (under models/): glue_ddaclae.py, snli.py, and sstb.py implement model training for each of the data sets considered in the submission. In particular, each file shows how model competency is estimated at the beginning of a training epoch, and how that estimate is used to select training data. Please note that snli.py implements experiments for all two-sentence GLUE classification tasks and sstb.py implements experiments for all single-sentence GLUE classification tasks.

2. ability estimation: irt_scoring.py implements the ability estimation given a data set with known difficulties.

3. Data selection: build_features.py, specifically the get_epoch_training_data and get_epoch_training_data_vision methods, implement the data selection code given a task and an estimated ability level, as well as several baselines.

Package prerequisites include:

- Dynet 2.0
- HuggingFace transformers
- PyTorch
- scikit-learn
- numpy
- pandas
- scipy 

Below are example python scripts to run the experiments:

# BERT models:

NUMEPOCHS=10  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=~/data/artificial-crowd-rps  # where the learned difficulty parameters are saved
DATADIR=~/data/glue  # where the GLUE data is saved
MINTRAINLEN=1000
TASK=$1  # set this to one of the GLUE tasks
CACHEDIR=~/data/bert/  # where you want to download the pre-trained BERT model


# set $STRATEGY equal to one of the following: baseline, ordered, simple, theta, naacl-linear, naacl-root
# set $TASK equal to one of the following: MRPC QNLI RTE QQP MNLI

python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy $STRATEGY --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR --balanced


# LSTM models:

# set $STRATEGY equal to one of the following: 'baseline', 'ordered', 'simple', 'theta', 'naacl-linear', 'naacl-root' 
python -u -m models.sstb --dynet-autobatch 1 --dynet-gpus 1 --dynet-mem 10000 --gpu 0 --data-dir $DATADIR --strategy $STRATEGY --num-epochs $NUMEPOCHS --diff-dir $DIFFDIR --task SST-2 --use-length --balanced

# set $STRATEGY equal to one of the following: baseline, ordered, simple, theta, naacl-linear, naacl-root
# set $TASK equal to one of the following: MRPC QNLI RTE QQP MNLI
python -u -m models.snli --dynet-autobatch 1 --dynet-gpus 1 --dynet-mem 10000 --gpu 0 --data-dir $DATADIR --strategy $STRATEGY --num-epochs $NUMEPOCHS --diff-dir $DIFFDIR --task $TASK --use-length --balanced 


