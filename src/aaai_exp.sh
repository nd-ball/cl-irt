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
