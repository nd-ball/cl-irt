#!/bin/bash

#$ -q gpu
#$ -l gpu_card=1
#$ -N ddaclae-test

# 1 - num supervised
# 2 - run number (for separating logs and rps)
# 3 - number of batches with IRT training
# 4 - number of VI epochs per training epoch (they're different for now)

module load conda
module load cuda
module load cudnn
source activate ddaclae

NUMEPOCHS=10  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=~/data/artificial-crowd-rps
DATADIR=~/data/glue
MINTRAINLEN=1000
TASK=MNLI
CACHEDIR=~/data/bert/

python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR

