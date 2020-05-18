#!/bin/bash

#$ -q gpu
#$ -l gpu_card=1
#$ -N lstm-single-sent

module load conda
module load cuda
module load cudnn
source activate ddaclae-lstm

# export LD_LIBRARY_PATH=/afs/crc.nd.edu/user/j/jlalor1/.conda/envs/ddaclae-lstm/lib/:$LD_LIBRARY_PATH

NUMEPOCHS=100  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=~/data/artificial-crowd-rps
DATADIR=~/data/glue
MINTRAINLEN=1000
TASK=$1
CACHEDIR=~/data/bert/

#python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR

python -u -m models.sstb --dynet-autobatch 1 --dynet-gpus 1 --dynet-mem 2000 --gpu 0 --data-dir $DATADIR --strategy theta --num-epochs $NUMEPOCHS --diff-dir $DIFFDIR --task $TASK 
