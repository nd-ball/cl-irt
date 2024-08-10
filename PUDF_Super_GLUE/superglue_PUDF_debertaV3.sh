#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 4
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe

#$ -N superglue_PUDF_deberta

source cl.env
NUMEPOCHS=20  # max num epochs, using early stopping though
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta
DIFFDIR=./difficulty
DATADIR=~/data/super_glue
MINTRAINLEN=100
TASK=wsc
CACHEDIR=~/data/debertav3/

python PUDF_superglue_DebertaV3.py --gpu 0 --data-dir $DATADIR --strategy theta \
--min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR \
--task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR
#python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR --balanced

