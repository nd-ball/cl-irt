#!/bin/bash

#$ -q gpu
#$ -l gpu_card=1
#$ -N lstm-two-sent

module load conda
module load cuda
module load cudnn
source activate ddaclae-lstm

NUMEPOCHS=100  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=~/data/artificial-crowd-rps
DATADIR=~/data/glue
MINTRAINLEN=1000
TASK=$1
STRATEGY=$2
LENGTH=$3
CACHEDIR=~/data/bert/

if ["$LENGTH" = "TRUE"]; then
python -u -m models.snli --dynet-autobatch 1 --dynet-gpus 1 --dynet-mem 10000 --gpu 0 --data-dir $DATADIR --strategy $STRATEGY --num-epochs $NUMEPOCHS --diff-dir $DIFFDIR --task $TASK --use-length 
else
python -u -m models.snli --dynet-autobatch 1 --dynet-gpus 1 --dynet-mem 10000 --gpu 0 --data-dir $DATADIR --strategy $STRATEGY --num-epochs $NUMEPOCHS --diff-dir $DIFFDIR --task $TASK 
fi


echo $TASK-$STRATEGY-$LENGTH 