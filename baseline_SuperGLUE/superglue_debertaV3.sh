#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 4
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N superglue_deberta_2

source cl.env
python baseline_SuperGLUE_debertaV3.py
