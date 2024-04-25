import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

output_csv_file = 'model_performance_summary.csv'

if not os.path.exists(output_csv_file):
    tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]


    # Initialize dictionaries
    difficulties = {}

    # Load difficulties from JSON files
    for task in tasks:
        with open(f"./diff_w_ft/3iter/{task}1pl/best_parameters.json", "r") as file:
            data = json.load(file)
            difficulties[task] = data["diff"]

    results_directory = 'results_w_finetuning/results_3'

    tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]

    with open("train_labels.json") as json_file:
        train_labels = json.load(json_file)

    results_directory = 'results_w_finetuning/results_3'

    data_for_csv = []
    # Iterate through the JSON files in the directory
    for filename in os.listdir(results_directory):
        if filename.endswith('.json'):
            # Extract information from the filename
            parts = filename.split('_')
            model_name = parts[0]
            task_name = parts[1]
            with open(os.path.join(results_directory, filename), 'r') as json_file:
                data = json.load(json_file)
            logits = data.get('logits')
            labels= train_labels[task_name]
            selected_logits = [logits[i][label] for i, label in enumerate(labels)]
            diff=difficulties[task_name]
            data_for_csv.append({
                "task_name": task_name,
                "model_name": model_name,
                "diff": diff,
                "selected_logits": selected_logits  # This will be a list; consider how you want to represent it in CSV
            })
    df = pd.DataFrame(data_for_csv)
    print(df.head())

    output_csv_file = 'model_performance_summary.csv'
    df.to_csv(output_csv_file, index=False)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
file_path = 'model_performance_summary.csv'
df = pd.read_csv(file_path, converters={'selected_logits': eval, 'diff': eval})


def confidence_interval(X, y, reg, X_pred, confidence=0.95):
    y_pred = reg.predict(X_pred)

    # Calculate the predictions' standard error (simplified)
    # This is a placeholder for the correct statistical calculation
    # Adjust this calculation based on your specific needs
    se = np.sqrt(mean_squared_error(y, reg.predict(X))) / np.sqrt(len(y))

    # Calculate the degrees of freedom
    df = len(y) - 2

    # Calculate the t multiplier for the given confidence level
    t_multiplier = t.ppf((1 + confidence) / 2., df)

    # Calculate the margin of error (simplified version)
    # Adjust this to match your statistical model
    margin_error = se * t_multiplier

    # Calculate confidence intervals
    lower_bound = y_pred - margin_error*0.5
    upper_bound = y_pred + margin_error*0.5

    return lower_bound, upper_bound


# Define tasks and models
tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]
models = [
        "bert-base-uncased",

          "distilbert-base-uncased",
          "roberta-base",
         'deberta-base',
          "albert-base-v2",
          "xlnet-base-cased",
          "electra-base-discriminator",
          "t5-base",
          "bart-base",
           "gpt2"]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
file_path = 'model_performance_summary.csv'
df = pd.read_csv(file_path, converters={'selected_logits': eval, 'diff': eval})



# Adjust figure size and font size
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24, 14))
axs = axs.flatten()

legend_handles = []
legend_labels = []

for task_index, task in enumerate(tasks):
    task_df = df[df['task_name'] == task]

    for model in models:
        model_df = task_df[task_df['model_name'] == model]

        if not model_df.empty:
            diffs = np.concatenate(model_df['diff'].to_numpy())
            logits = np.concatenate(model_df['selected_logits'].to_numpy())

            # Ensure the diffs are within the desired range
            valid_indices = (diffs >= -5) & (diffs <= 5)
            diffs = diffs[valid_indices]
            logits = logits[valid_indices]

            # Assuming you're within the loop for plotting each task and model
            if len(diffs) > 1 and len(logits) > 1:
                X = diffs.reshape(-1, 1)
                y = logits
                reg = LinearRegression().fit(X, y)

                X_pred = np.linspace(-5, 5, 100).reshape(-1, 1)
                lower_ci, upper_ci = confidence_interval(X, y, reg, X_pred, confidence=0.95)
                # print(X_pred.flatten().shape, lower_ci.shape, upper_ci.shape)  # This should now print matching shapes

                # Now, plotting with fill_between should work without broadcasting issues
                axs[task_index].plot(X_pred, reg.predict(X_pred), label=model, linewidth=3)
                axs[task_index].fill_between(X_pred.flatten(), lower_ci, upper_ci, alpha=0.2)

    axs[task_index].set_xlim([-5, 5])
    axs[task_index].set_title(task)
    axs[task_index].grid(True)

# Manually set common x and y labels
fig.text(0.5, 0.01, 'Diff', ha='center')
fig.text(0.01, 0.5, 'Selected Logits', ha='center', rotation='vertical')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=5)

plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
plt.show()


