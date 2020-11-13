
for n in 1 2 3 4 #5
do
for TASK in MRPC QNLI RTE QQP MNLI SST-2 
do 
    # ddaclae
    qsub glue_test_crc.sh $TASK 

    # baseline
    qsub bert_baseline_crc.sh $TASK 

    # CBCL Linear Heuristic
    qsub bert_cbcl_linear_heuristic.sh $TASK 

    # CBCL Linear IRT Diff
    qsub bert_cbcl_linear_irt.sh $TASK 

    # CBCL Root Heuristic
    qsub bert_cbcl_root_heuristic.sh $TASK 

    # CBCL Root IRT Diff
    qsub bert_cbcl_root_irt.sh $TASK 
done 
sleep 20
done
