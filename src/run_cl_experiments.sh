# run curriculum learning experiments on gypsum
# setup
#export LD_LIBRARY_PATH=/home/lalor/bin/dynet-base-py3/dynet/build/dynet/:$LD_LIBRARY_PATH

NUMEPOCHS=100

# baselines using text length (just premise for SNLI)
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_baseline_length_not_balanced_easiest.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --use-length --ordering easiest --num-epochs $NUMEPOCHS" 

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_baseline_length_not_balanced_hardest.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --use-length --ordering hardest --num-epochs $NUMEPOCHS"

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_baseline_length_not_balanced_middleout.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --use-length --ordering middleout --num-epochs $NUMEPOCHS"


sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_baseline_length_not_balanced_easiest.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --use-length --ordering easiest --num-epochs $NUMEPOCHS"

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_baseline_length_not_balanced_hardest.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --use-length --ordering hardest --num-epochs $NUMEPOCHS"

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_baseline_length_not_balanced_middleout.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --use-length --ordering middleout --num-epochs $NUMEPOCHS"

# SNLI 
# baseline 
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_baseline.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --num-epochs $NUMEPOCHS"

for o in easiest middleout hardest
do
    for s in simple #balanced 
    do 
        # CL, simple, balanced
        sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_balanced-$s-$o.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --balanced --ordering $o --num-epochs $NUMEPOCHS"

        # CL, simple, not balanced
        sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_not_balanced-$s-$o.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --ordering $o --num-epochs $NUMEPOCHS"
    done 
done 


# SSTB 
# baseline 
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_baseline_easiest.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --num-epochs $NUMEPOCHS"

for o in easiest middleout hardest
do
    for s in simple #balanced 
    do 
        # CL, simple, balanced
        sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb__cl_balanced-$s-$o.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --balanced --ordering $o --num-epochs $NUMEPOCHS"

        # CL, simple, not balanced
        sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_not_balanced-$s-$o.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --ordering $o --num-epochs $NUMEPOCHS"
    done 
done 



##### Add random baselines (4 per data set) #####
### SNLI
# random ordered (fixed order across all epochs) 
# not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_ordered_not_balanced_hardest_random.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --ordering hardest --random  --num-epochs $NUMEPOCHS"

# balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_ordered_balanced_hardest_random.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced --ordering hardest --random  --num-epochs $NUMEPOCHS"

# random simple 
# not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli_cl_simple_not_balanced_hardest_random.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --ordering hardest --random --num-epochs $NUMEPOCHS"

# balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/snli__cl_simple_balanced_hardest_random.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced --ordering hardest --balanced --num-epochs $NUMEPOCHS"



### SSTB
# random ordered (fixed order across all epochs) 
# balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_ordered_balanced_hardest_random.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced --ordering hardest --random --num-epochs $NUMEPOCHS"

# not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_ordered_not_balanced_hardest_random.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --ordering hardest --random --num-epochs $NUMEPOCHS"


# random simple 
# balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb__cl_simple_balanced_hardest_random.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced --ordering hardest --random --num-epochs $NUMEPOCHS"

# not balanced
sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/sstb_cl_simple_not_balanced_hardest_random.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --ordering hardest --random --num-epochs $NUMEPOCHS" 

# irt CL

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/irt_cl_snli.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS"

sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/irt_cl_sstb.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS"
