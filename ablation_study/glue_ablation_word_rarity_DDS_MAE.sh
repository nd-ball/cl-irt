#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 4
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe

#$ -N glue_word_rarity_DDS_MAE_2  # Job name

source cl.env
NUMEPOCHS=20  # max num epochs, using early stopping though
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=../PUDF_GLUE/difficulty
DATADIR=~/data/glue
MINTRAINLEN=1000
TASK=mrpc
CACHEDIR=~/data/debertav3/

python  ablation_GLUE_DebertaV3_predefine_PUDF.py --gpu 0 --data-dir $DATADIR \
--difficulty_measurer word_rarity --training_scheduler DDS_MAE \
--min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR \
--num-obs $NUMOBS --diff-dir $DIFFDIR
#python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR --balanced

