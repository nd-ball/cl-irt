#!/bin/bash

#$ -q gpu
#$ -l gpu_card=1
#$ -N cbcl-linear-irt

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

python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy naacl-linear --ordering easiest --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR
python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy naacl-linear --ordering easiest --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR --balanced 
