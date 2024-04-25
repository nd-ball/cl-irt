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
GLUE_TASKS = [ "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2"]

#read one jason file in results folder
# with open('results/albert-base-v2_mnli_128_response_logits_Accuracy_0.33432730161802077.json') as f:
#     data = json.load(f)
# print(data.keys())



# Want to craete a .csv files with columns:
# 1. the first column is the model name and sequence length, e.g.,albert-base-v2_128
# 2. the second column is the task name, e.g., mnli
# 3. the third column is the data ID from 0,1,2...until the end of the dataset
# 4. the fourth column is the response
# The first first two columns are obtained from the file name, the third column is the index of the data in the jason file, and the fourth column is the responses from each jason file.
# can you help me to write a python script to do this?

results_directory = 'results_w_finetuning/results_3'
# results_directory = 'results_wo_finetuning'

# Iterate through the JSON files in the directory
for filename in os.listdir(results_directory):
    if filename.endswith('.json'):
        # Extract information from the filename
        parts = filename.split('_')
        model_name = parts[0]
        task_name = parts[1]
        sequence_length = parts[2]
        modelID= model_name+'_'+sequence_length

        # Load the JSON file and extract the response
        with open(os.path.join(results_directory, filename), 'r') as json_file:
            data = json.load(json_file)
        response = data.get('responses')
        itemID = range(len(response))
        csv_filename = f"{task_name}.csv"
        if not os.path.exists(csv_filename):

            df = pd.DataFrame({
                'modelID': [modelID] * len(response),  # Repeat the modelID for each response
                'itemID': itemID,
                'response': response
            })
            df.to_csv(csv_filename, index=False)
        else:
            # Create a DataFrame to append to the CSV
            df = pd.DataFrame({
                'modelID': [modelID] * len(response),  # Repeat the modelID for each response
                'itemID': itemID,
                'response': response
            })

            df.to_csv(csv_filename, mode='a', index=False, header=False)
