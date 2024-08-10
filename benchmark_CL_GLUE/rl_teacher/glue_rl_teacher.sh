#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 4
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe

#$ -N glue_rl_teacher_deberta_2

source rl_teacher.env
python rl_teacher.py