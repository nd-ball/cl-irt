#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 4
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe

#$ -N glue_transfer_teacher_deberta_2

source transfer_teacher.env
python transfer_teacher_GLUE_deberta.py
