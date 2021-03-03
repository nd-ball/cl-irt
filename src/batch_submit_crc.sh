

# submit based on theta
for lb in -6 -5 -4 -3 -2.5 -2 -1.5 -1 -0.5
do
  for ub in 3 2.5 2 1.5 1 0.5
  do
    for TASK in MRPC QNLI RTE QQP MNLI SST-2 
    do 
    # ddaclae
    qsub glue_test_crc.sh $TASK $lb $ub 

    # baseline
    #qsub bert_baseline_crc.sh $TASK 

    # CBCL Linear Heuristic
    #qsub bert_cbcl_linear_heuristic.sh $TASK 

    # CBCL Linear IRT Diff
    #qsub bert_cbcl_linear_irt.sh $TASK 

    # CBCL Root Heuristic
    #qsub bert_cbcl_root_heuristic.sh $TASK 

    # CBCL Root IRT Diff
    #qsub bert_cbcl_root_irt.sh $TASK 
done 
#sleep 10
done
sleep 10
done 
