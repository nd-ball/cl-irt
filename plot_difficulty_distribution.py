import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]
sns.set(style="whitegrid")
# models = [
#         "bert-base-uncased",
#           "distilbert-based-uncased",
#           "roberta-base",
#          'deberta-base',
#           "albert-base-v2",
#           "xlnet-base-cased",
#           "electra-base-discriminator",
#           "t5-base",
#           "bart-base",
#            "gpt2"]

# Modified plot function
def plot_hist(data, task, ax):
    ax.hist(data, bins=60, edgecolor='black', color='skyblue')

    # Calculate statistics
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)

    # Annotate the subplot with statistics
    stats_text = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Add titles and labels
    ax.set_title(f'{task} Difficulty Distribution', fontsize=12)
    ax.set_xlabel('Difficulty value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Create a figure with 2 rows and 3 columns of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten() # Flatten the array of axes

for idx, task in enumerate(tasks):
    with open(f"diff_w_ft/3iter/{task}1pl/best_parameters.json", "r") as file:
    # with open(f"diff_wo_ft/{task}1pl/best_parameters.json", "r") as file:
        data = json.load(file)
        plot_hist(data["diff"], task, axes[idx])

# Adjust layout
plt.tight_layout()
plt.show()


