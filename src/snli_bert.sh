
NUMEPOCHS=100  # max num epochs, using early stopping though 

#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/bert/bert-snli-ddaclae-test.log --wrap="python -u -m models.snli_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 1000 --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"

#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/bert/bert-sstb-ddaclae-test.log --wrap="python -u -m models.sstb_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy theta --min-train-length 1000 --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"

#for m in sstb_bert snli_bert 
#do 
#    for strat in naacl-linear naacl-root 
#    do 
#        for o in easiest hardest  
#        do 
#            sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/bert/bert-$m-$strat-$o-length.log --wrap="python -u -m models.$m --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $strat --use-length --ordering $o --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"
#            sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/bert/bert-$m-$strat-$o-irt.log --wrap="python -u -m models.$m --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy $strat --ordering $o --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/" 
#        done 
#    done
#done


sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/bert/bert-sstb-test-length.log --wrap="python -u -m models.sstb_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy naacl-linear --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"

#sbatch -p titanx-long --gres=gpu:1 --mem=90gb --output=logs/bert/bert-snli-test-length.log --wrap="python -u -m models.snli_bert --gpu 0 --data-dir /mnt/nfs/work1/hongyu/lalor/data/cl-data/ --strategy naacl-linear --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir /mnt/nfs/work1/hongyu/lalor/data/bert/"
