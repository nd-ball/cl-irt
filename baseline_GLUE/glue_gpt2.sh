#!/bin/bash
#$ -q gpu@@csecri-v100
#$ -l gpu_card=1
#$ -pe smp 4
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N glue_gpt2_2

source cl.env
python baseline_GLUE_gpt2.py
