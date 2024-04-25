import numpy as np
# from sklearn.metrics import calibration_curve
import pandas as pd
import json
import os
import os
import json
import pandas as pd
from pathlib import Path

models = [
        "bert-base-uncased",
          "distilbert-based-uncased",
          "roberta-base",
         'deberta-base',
          "albert-base-v2",
          "xlnet-base-cased",
          "electra-base-discriminator",
          "t5-base",
          "bart-base",
           "gpt2"]
#
# GLUE_TASKS = ["cola", "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
GLUE_TASKS = [  "mnli","mrpc", "qnli", "qqp", "rte", "sst2"]

def expected_calibration_error(samples, accuracies, M=20):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    # predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    # accuracies = predicted_label==true_labels
    accuracies  = np.array([bool(x) for x in accuracies ])
    # print(accuracies)
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece



results_directory = 'results_w_finetuning/results_3'
# results_directory = 'results_wo_finetuning'
for filename in os.listdir(results_directory):
    if filename.endswith('.json'):
        # Extract information from the filename
        parts = filename.split('_')
        model_name = parts[0]
        task_name = parts[1]
        sequence_length = parts[2]
        modelID= model_name+'_'+sequence_length
        print("=======")
        print(modelID)
        print(task_name)
        with open(os.path.join(results_directory, filename), 'r') as json_file:
            data = json.load(json_file)
        accuracy = data.get('responses')
        logits = data.get('logits')
        ece=expected_calibration_error(logits, accuracy)
        print(ece)



