import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

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
# Initialize dictionaries
difficulties = {}
model_accuracies = {task: {} for task in tasks}
# Redefine bin_ranges according to the new specification
bin_ranges = np.arange(-4, 5, 1, dtype=float)  # Creates bins from -4 to 4, step 1, as float
# Append minimum and maximum to the bin_ranges for covering the full range
bin_ranges = np.insert(bin_ranges, 0, -np.inf)  # Append -inf at the start
bin_ranges = np.append(bin_ranges, np.inf)
import numpy as np


def bin_difficulties_and_calculate_percentage(difficulties, responses):
    bins_edges = np.array([-np.inf] + list(range(-3, 4)) + [np.inf])
    correct_counts = np.zeros(len(bins_edges) - 1)
    total_counts = np.zeros(len(bins_edges) - 1)

    # Bin difficulties
    binned_difficulties = np.digitize(difficulties, bins_edges) - 1

    # Count responses
    for bin_idx, response in zip(binned_difficulties, responses):
        total_counts[bin_idx] += 1
        if response == 1:  # Assuming 1 means correct response
            correct_counts[bin_idx] += 1

    # Calculate percentages
    percentage_correct = (correct_counts / total_counts) * 100
    percentage_correct = np.nan_to_num(percentage_correct)

    return percentage_correct, bins_edges


# Load difficulties from JSON files
for task in tasks:
    with open(f"./diff_w_ft/3iter/{task}1pl/best_parameters.json", "r") as file:
        data = json.load(file)
        difficulties[task] = data["diff"]




# Assuming the function bin_difficulties_and_calculate_percentage is defined as before

# Process CSV files and store accuracy percentages and bin ranges
for task in tasks:
    df = pd.read_csv(f"{task}.csv")
    for model in df["modelID"].unique():
        model_name_simplified = model.split('_')[0]
        model_responses = df[df['modelID'] == model]['response']
        model_accuracies[task][model_name_simplified], _ = bin_difficulties_and_calculate_percentage(
            difficulties[task], model_responses)

# Plotting
num_bins = 8  # Adjust based on the actual number of bins from bin_difficulties_and_calculate_percentage
gap_between_ranges=0.02
# Plotting, revised to avoid index out-of-bounds errors
fig, axs = plt.subplots(2, 3, figsize=(36, 24), sharey=True)
axs = axs.flatten()

for i, task in enumerate(tasks):
    ax = axs[i]
    num_models = len(model_accuracies[task].keys())
    total_width = 1 - gap_between_ranges * (num_bins - 1)
    bar_width = total_width / (num_models * num_bins)

    for model_index, (model, acc_percentage) in enumerate(model_accuracies[task].items()):
        # Ensure acc_percentage has the correct length matching num_bins
        if len(acc_percentage) != num_bins:
            raise ValueError(f"Accuracy percentage array for model '{model}' in task '{task}' does not match the expected number of bins ({num_bins}).")
        for bin_index in range(num_bins):
            position = (bin_index * (num_models * bar_width + gap_between_ranges)) + model_index * bar_width
            ax.bar(position, acc_percentage[bin_index], bar_width, label=model if i == 0 and bin_index == 0 else "")

    bin_labels = ['<-3'] + [f'[{i},{i+1}]' for i in range(-3, 3)] + ['>3']
    ax.set_xticks([i * (num_models * bar_width + gap_between_ranges) + (num_models - 1) * bar_width / 2 for i in range(num_bins)])
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax.set_title(task)
    ax.grid(True)

fig.text(0.5, 0.001, 'Difficulty Range', ha='center', va='bottom', fontsize=18)  # X-axis label moved lower
fig.text(0.0001, 0.5, 'Percentage of Correct Responses', va='center', rotation='vertical', fontsize=18)  # Y-axis label moved lefter

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=10,bbox_to_anchor=(0.5, 0.985), fontsize=16)  # Legend moved higher and font size increased

plt.tight_layout()
plt.show()