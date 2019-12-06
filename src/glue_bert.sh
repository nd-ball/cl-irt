
NUMEPOCHS=10  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=/mnt/nfs/work1/hongyu/lalor/data/jiant/artificial-crowd-generation-2/rps
DATADIR=/mnt/nfs/work1/hongyu/lalor/data/glue
MINTRAINLEN=100
TASK=QNLI
CACHEDIR=/mnt/nfs/work1/hongyu/lalor/data/bert/


for TASK in CoLA MRPC QNLI RTE WNLI QQP
do 


    # DDaCLAE
    sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/v2/bert-$TASK-ddaclae-test-%j.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR"


    # NAACL Baselines

    
    for strat in naacl-linear naacl-root 
    do 
        for o in easiest hardest  
        do 
            sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/v2/bert-$TASK-$strat-$o-length.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy $strat --use-length --ordering $o --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR"
            sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/v2/bert/bert-$TASK-$strat-$o-irt.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy $strat --ordering $o --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR" 
        done 
    done

    # Fully supervised baselines
    sbatch -p titanx-long --gres=gpu:1 --mem=64gb --output=logs/bert/v2/bert-$TASK-baseline.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy baseline --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR"
done 
