# list of aaai experiments 

NUMEPOCHS=200  # max num epochs, using early stopping though 

# mnist/cifar 

for m in mnist cifar 
do 
    # baselines 
    sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/$m-baseline.log --wrap="python -u -m models.$m --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy baseline"

    # irt CL
    sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/irt-cl-$m-1000.log --wrap="python -u -m models.$m --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta --ordering easiest --min-train-length 1000"

    # irt CL (hard)
    sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/irt-cl-hard-$m-1000.log --wrap="python -u -m models.$m --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy theta-hard --ordering hardest --min-train-length 1000"

    # NAACL Baselines 
    # linear 
    for strat in naacl-linear naacl-root 
    do 
        for o in easiest hardest middleout 
        do 
            sbatch -p m40-long --gres=gpu:1 --mem=90gb --output=logs/$m-$strat-$o.log --wrap="python -u -m models.$m --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --num-epochs $NUMEPOCHS --strategy $strat --ordering $o"
        done 
    done
done 

# nlp tasks 
for m in sstb snli 
do 
    # baselines 
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/$m-baseline.log --wrap="python -u -m models.$m --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy baseline --num-epochs $NUMEPOCHS"

    # irt CL 
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/irt-cl-$m.log --wrap="python -u -m models.$m --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 500 --num-epochs $NUMEPOCHS"

    # irt CL (hard) 
    sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/irt-cl-$m.log --wrap="python -u -m models.$m --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --min-train-length 500 --num-epochs $NUMEPOCHS --strategy theta-hard --ordering hardest"

    # NAACL Baselines 
    for strat in naacl-linear naacl-root 
    do 
        for o in easiest hardest middleout 
        do 
            sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/$m-$strat-$o-length.log --wrap="python -u -m models.$m --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $strat --use-length --ordering $o --num-epochs $NUMEPOCHS"
            sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/$m-$strat-$o-irt.log --wrap="python -u -m models.$m --dynet-autobatch 1 --dynet-gpu 1 --dynet-mem 11000 --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $strat --ordering $o --num-epochs $NUMEPOCHS" 
        done 
    done 
done 


