# run curriculum learning experiments on gypsum
# setup
#export LD_LIBRARY_PATH=/home/lalor/bin/dynet-base-py3/dynet/build/dynet/:$LD_LIBRARY_PATH

# SNLI 
# baseline 
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_baseline.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline"

# CL, simple, balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli__cl_simple_balanced.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced"

# CL, simple, not balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_simple_not_balanced.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple"

# CL, ordered, balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_ordered_balanced.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced"

# CL, ordered, not balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_ordered_not_balanced.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered"

# SSTB 
# baseline 
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_baseline.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline"

# CL, simple, balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb__cl_simple_balanced.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced"

# CL, simple, not balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_simple_not_balanced.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple"

# CL, ordered, balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_ordered_balanced.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced"

# CL, ordered, not balanced
#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_ordered_not_balanced.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered"


##### same as above, but with ordering = middleout ####
# SNLI 
# CL, simple, balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli__cl_simple_balanced_middleout.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced --ordering middleout"

# CL, simple, not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_simple_not_balanced_middleout.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --ordering middleout"

# CL, ordered, balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_ordered_balanced_middleout.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced --ordering middleout"

# CL, ordered, not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_ordered_not_balanced_middleout.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --ordering middleout"

# SSTB 
# CL, simple, balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb__cl_simple_balanced_middleout.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced --ordering middleout"

# CL, simple, not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_simple_not_balanced_middleout.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --ordering middleout" 

# CL, ordered, balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_ordered_balanced_middleout.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced --ordering middleout"

# CL, ordered, not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_ordered_not_balanced_middleout.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --ordering middleout"
