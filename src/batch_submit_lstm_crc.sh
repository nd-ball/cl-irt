
for n in 1 2 3 4 5
do

# sentiment
qsub lstm_single_sent.sh SST-2 theta FALSE
qsub lstm_single_sent.sh SST-2 baseline FALSE
qsub lstm_single_sent.sh SST-2 naacl-linear FALSE 
qsub lstm_single_sent.sh SST-2 naacl-root FALSE
qsub lstm_single_sent.sh SST-2 naacl-linear TRUE
qsub lstm_single_sent.sh SST-2 naacl-root TRUE

# sentiment will be separate
for TASK in MRPC QNLI RTE QQP MNLI 
do 
    # baseline', 'ordered', 'simple', 'theta', 'naacl-linear', 'naacl-root' 
    # ddaclae
    qsub lstm_two_sent.sh $TASK theta FALSE
    # baseline
    qsub lstm_two_sent.sh $TASK baseline FALSE
    # CBCL Linear Heuristic
    qsub lstm_two_sent.sh $TASK naacl-linear TRUE
    # CBCL Linear IRT Diff
    qsub lstm_two_sent.sh $TASK naacl-linear FALSE
    # CBCL Root Heuristic
    qsub lstm_two_sent.sh $TASK naacl-root TRUE
    # CBCL Root IRT Diff
    qsub lstm_two_sent.sh $TASK naacl-root FALSE
done 
sleep 20
done