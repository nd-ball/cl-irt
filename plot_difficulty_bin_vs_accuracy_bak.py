import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]

# Initialize dictionaries
difficulties = {}
model_accuracies = {task: {} for task in tasks}
bin_ranges = {task: None for task in tasks}  # To store bin ranges for each task

# Load difficulties from JSON files
for task in tasks:
    with open(f"./diff_w_ft/3iter/{task}1pl/best_parameters.json", "r") as file:
        data = json.load(file)
        difficulties[task] = data["diff"]


def bin_difficulties_and_calculate_percentage(difficulties, responses, num_bins=10):
    bins = np.linspace(min(difficulties), max(difficulties), num_bins + 1)
    binned_difficulties = np.digitize(difficulties, bins) - 1
    binned_difficulties = np.clip(binned_difficulties, 0, num_bins - 1)

    correct_counts = np.zeros(num_bins)
    total_counts = np.zeros(num_bins)

    for bin_idx, response in zip(binned_difficulties, responses):
        total_counts[bin_idx] += 1
        if response == 1:
            correct_counts[bin_idx] += 1

    correct_counts = np.nan_to_num(correct_counts)
    total_counts = np.nan_to_num(total_counts)

    percentage_correct = np.divide(correct_counts, total_counts, where=total_counts > 0) * 100
    percentage_correct = np.nan_to_num(percentage_correct)

    return percentage_correct, bins,total_counts


# Process CSV files and store accuracy percentages and bin ranges
for task in tasks:
    df = pd.read_csv(f"{task}.csv")
    for model in df["modelID"].unique():
        model_name_simplified = model.split('_')[0]
        model_responses = df[df['modelID'] == model]['response']
        model_accuracies[task][model_name_simplified], bins,total_counts = bin_difficulties_and_calculate_percentage(
            difficulties[task], model_responses)
    bin_ranges[task] = bins  # Store bin ranges for each task


# Assuming all other code remains the same and leads up to this point...

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(20, 17), sharey=True)
axs = axs.flatten()

# Define a larger font size for axis labels and tick labels
axis_label_fontsize = 14  # For x-axis and y-axis labels
tick_label_fontsize = 10  # For tick labels

for i, task in enumerate(tasks):
    ax = axs[i]
    bins = bin_ranges[task]  # Retrieve bin ranges for the task
    num_models = len(model_accuracies[task].keys())

    # Calculate bar width and offsets dynamically based on the number of models
    bar_width_factor = 1  # Adjust as needed to make bars wider or narrower
    bar_width = 1 / (num_models + bar_width_factor)
    offsets = np.linspace(-bar_width * num_models / 2, bar_width * num_models / 2, num_models)

    model_index = 0
    active_bins = []  # To track bins with predictions (total_counts > 0)
    for model, acc_percentage in model_accuracies[task].items():
        _, bins_used,total_counts = bin_difficulties_and_calculate_percentage(difficulties[task],
                                                                 df[df['modelID'] == model.split('_')[0]]['response'])
        positions = np.arange(len(acc_percentage)) + offsets[model_index]

        # Plot bars only for bins with predictions
        for pos, acc, bin_start, bin_end, count in zip(positions, acc_percentage, bins_used[:-1], bins_used[1:],total_counts):
            if acc > 0:  # Check if there are predictions and corrections
                ax.bar(pos, acc, bar_width, label=model if model_index == 0 else "")
                if model_index == 0:  # Track active bins for labeling
                    active_bins.append(f"[{bin_start:.2f}, {bin_end:.2f}]")
            elif acc == 0:  # Check if there are predictions but no corrections
                ax.bar(pos, 0, bar_width, label=model if model_index == 0 else "", color='grey')
                if model_index == 0:
                    active_bins.append(f"[{bin_start:.2f}, {bin_end:.2f}] (n={count})")

        model_index += 1

    ax.set_title(task)
    ax.grid(True)
    ax.set_xticks(np.arange(len(active_bins)))
    ax.set_xticklabels(active_bins, rotation=45, ha="right", fontsize=tick_label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)

# Adjustments for the axis labels and legend
fig.text(0.5, 0.02, 'Difficulty Range', ha='center', fontsize=axis_label_fontsize)
fig.text(0.02, 0.5, 'Percentage of Correct Responses', ha='center', rotation='vertical', fontsize=axis_label_fontsize)
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(labels) // 2,
           fontsize=axis_label_fontsize)

plt.show()