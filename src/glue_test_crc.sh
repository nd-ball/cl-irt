#!/bin/bash

#$ -q gpu
#$ -l gpu_card=1
#$ -pe smp 1
#$ -N ddaclae

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
TASK=$1
CACHEDIR=~/data/bert/
LB=$2
UB=$3

python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR --lower-bound $LB --upper-bound $UB
#python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR --balanced --upper-bound $UB --lower-bound $LB

