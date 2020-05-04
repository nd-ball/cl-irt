# run curriculum learning experiments on gypsum
# setup
#export LD_LIBRARY_PATH=/home/lalor/bin/dynet-base-py3/dynet/build/dynet/:$LD_LIBRARY_PATH

NUMEPOCHS=200
for k in 1 2 3 4 5 
do 
    # SNLI 
    # baseline 
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli_baseline-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --num-epochs $NUMEPOCHS"

    for o in easiest middleout hardest
    do
        for s in simple ordered #balanced 
        do 
            # CL, balanced
            sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli_cl_balanced-$s-$o-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --balanced --ordering $o --num-epochs $NUMEPOCHS"

            # CL, not balanced
            sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli_cl_not_balanced-$s-$o-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --ordering $o --num-epochs $NUMEPOCHS"
        done 
    done 


    # SSTB 
    # baseline 
    sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb_baseline-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --num-epochs $NUMEPOCHS"

    for o in easiest middleout hardest
    do
        for s in simple ordered #balanced 
        do 
            # CL, balanced
            sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb_cl_balanced-$s-$o-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --balanced --ordering $o --num-epochs $NUMEPOCHS"

            # CL, not balanced
            sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb_cl_not_balanced-$s-$o-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $s --ordering $o --num-epochs $NUMEPOCHS"
        done 
    done 



    ##### Add random baselines (4 per data set) #####
    ### SNLI
    # random ordered (fixed order across all epochs) 
    # not balanced
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli_cl_ordered_not_balanced_hardest_random-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --ordering hardest --random  --num-epochs $NUMEPOCHS"

    # balanced
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli_cl_ordered_balanced_hardest_random-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced --ordering hardest --random  --num-epochs $NUMEPOCHS"

    # random simple 
    # not balanced
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli_cl_simple_not_balanced_random-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --ordering hardest --random --num-epochs $NUMEPOCHS"

    # balanced
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/snli__cl_simple_balanced_random-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced --ordering hardest --balanced --num-epochs $NUMEPOCHS"



    ### SSTB
    # random ordered (fixed order across all epochs) 
    # balanced
    sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb_cl_ordered_balanced_hardest_random-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --balanced --ordering hardest --random --num-epochs $NUMEPOCHS"

    # not balanced
    sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb_cl_ordered_not_balanced_hardest_random-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy ordered --ordering hardest --random --num-epochs $NUMEPOCHS"


    # random simple 
    # balanced
    sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb__cl_simple_balanced_random-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --balanced --ordering hardest --random --num-epochs $NUMEPOCHS"

    # not balanced
    sbatch -p titanx-short --gres=gpu:1 --mem=90gb --output=logs/robustness/sstb_cl_simple_not_balanced_random-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy simple --ordering hardest --random --num-epochs $NUMEPOCHS" 

    # irt CL

    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/irt_cl_snli-$k.log --wrap="python -u -m models.snli --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS"

    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/robustness/irt_cl_sstb-$k.log --wrap="python -u -m models.sstb --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS"
done 
