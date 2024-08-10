#!/bin/bash
#$ -M gmeng@nd.edu   # Email address for job notification
#$ -m abe
#$ -N difficulty_gen
qrsh -q gpu@@csecri-v100 -l gpu=1 -pe smp 8
export CUDA_VISIBLE_DEVICES=${SGE_HGR_gpu_card// /,}

conda activate cl
module load cuda/12.1
module load cudnn/8.9.3
python cl_base.py
