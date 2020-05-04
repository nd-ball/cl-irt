
NUMEPOCHS=10  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
LOGDIR=20191206-1

# DDaCLAE
#sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/bert-snli-ddaclae-test-%j.log --wrap="python -u -m models.snli_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 1000 --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"

sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-sstb-ddaclae-test-%j.log --wrap="python -u -m models.sstb_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 1000 --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"

# NAACL Baselines
#snli_bert
for m in sstb_bert snli_bert 
do 
    for strat in naacl-linear naacl-root 
    do 
        for o in easiest hardest  
        do 
            sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-$m-$strat-$o-length.log --wrap="python -u -m models.$m --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $strat --use-length --ordering $o --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/ --competency $COMP"
            sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-$m-$strat-$o-irt.log --wrap="python -u -m models.$m --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $strat --ordering $o --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/ --competency $COMP" 
        done 
    done
done

# Fully supervised baselines
sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-sstb-baseline.log --wrap="python -u -m models.sstb_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"

sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-snli-baseline.log --wrap="python -u -m models.snli_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"
