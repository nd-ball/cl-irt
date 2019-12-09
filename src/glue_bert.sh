
NUMEPOCHS=10  # max num epochs, using early stopping though 
COMP=5  # for baselines, competency at midpoint
NUMOBS=1000  # for estimating theta 
DIFFDIR=/mnt/nfs/work1/hongyu/lalor/data/jiant/artificial-crowd-generation-2/rps
DATADIR=/mnt/nfs/work1/hongyu/lalor/data/glue
MINTRAINLEN=1000
TASK=MNLI
CACHEDIR=/mnt/nfs/work1/hongyu/lalor/data/bert/
LOGDIR=20191206-naacl-2
QUEUE=titanx-short

for TASK in MRPC QNLI RTE WNLI QQP MNLI
#for TASK in MNLI CoLA 
do 
    # DDaCLAE
    sbatch -p $QUEUE --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-$TASK-ddaclae-test-%j.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy theta --min-train-length $MINTRAINLEN --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR"

    # Fully supervised baselines
    sbatch -p $QUEUE --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-$TASK-baseline.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy baseline --use-length --ordering easiest --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR"

    # NAACL Baselines    
    for strat in naacl-linear naacl-root 
    do 
        for o in easiest hardest  
        do 
            sbatch -p $QUEUE --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-$TASK-$strat-$o-length.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy $strat --use-length --ordering $o --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR"
            sbatch -p $QUEUE --gres=gpu:1 --mem=64gb --output=logs/bert/$LOGDIR/bert-$TASK-$strat-$o-irt.log --wrap="python -u -m models.glue_ddaclae --gpu 0 --data-dir $DATADIR --strategy $strat --ordering $o --num-epochs $NUMEPOCHS --cache-dir $CACHEDIR --competency $COMP --task $TASK --num-obs $NUMOBS --diff-dir $DIFFDIR" 
        done 
    done
done 
