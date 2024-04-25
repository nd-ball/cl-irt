import pandas as pd
import json
import os
#
GLUE_TASKS = [ "mnli",  "mrpc", "qnli", "qqp", "rte", "sst2"]
for task in GLUE_TASKS:
    print("=============================")
    D = pd.read_csv(f"{task}.csv")

    print(D.head())
    print(D.tail())

    response_patterns = {}

    for idx, row in D.iterrows():
        if row["modelID"] not in response_patterns:
            response_patterns[row["modelID"]] = {}
        response_patterns[row["modelID"]][f"q{row['itemID']}"] = row["response"]

    with open(f"{task}_pyirt.jsonlines", "w") as outfile:
        for key, val in response_patterns.items():
            outrow = {"subject_id": key, "responses": val}
            outfile.write(json.dumps(outrow) + "\n")

    # os.system(f"py-irt train 1pl {task}_pyirt.jsonlines --output {task}1pl")

