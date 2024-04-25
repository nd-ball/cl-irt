import numpy as np
import matplotlib.pyplot as plt
import json


# Define the sigmoid function
def sigmoid(b, theta):
    return 1 / (1 + np.exp(-(theta - b)))


tasks = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]

model_ability = {}
difficulties = {}

# Load the data from JSON files and fill the model_ability and difficulties dictionaries
for task in tasks:
    with open(f"./diff_w_ft/3iter/{task}1pl/best_parameters.json", "r") as file:
        data = json.load(file)
        task_ability = {}
        difficulties[task] = data["diff"]
        for subject_id, model_name in data["subject_ids"].items():
            model_name_simplified = model_name.split('_')[0]
            task_ability[model_name_simplified] = data["ability"][int(subject_id)]
        model_ability[task] = task_ability

# Define dataset names
dataset_names = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]

# Create a figure with 2 rows and 3 columns of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()  # Flatten the array of axes

# Plotting one line per model for each dataset in a separate subplot
lines = []  # Keep track of line objects for the legend
labels = []  # Keep track of label names for the legend
for i, dataset in enumerate(dataset_names):
    task_models = model_ability[dataset]
    task_difficulties = np.array(difficulties[dataset])
    plot_difficulties = np.linspace(0, 5, 100)

    for model_name, ability in task_models.items():
        predict_accuracy = sigmoid(plot_difficulties, ability)
        # Label each line with model name
        label = f'{model_name}'
        line, = axes[i].plot(plot_difficulties, predict_accuracy, label=label)

        if i == 0:  # Only add to legend list once
            lines.append(line)
            labels.append(label)

    axes[i].set_title(dataset)
    axes[i].set_xlabel('Difficulty')
    axes[i].set_ylabel('Predicted Accuracy')
    axes[i].grid(True)
    axes[i].label_outer()  # Hide x, y labels for inner plots

# Create a single legend for the whole figure
fig.legend(lines, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.92), fontsize='small')

# Adjust the layout and add a main title
plt.suptitle('Predicted Accuracy vs Difficulty for Different Datasets and Models', fontsize=16)
# fig.subplots_adjust(top=0.85)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the padding to provide space for the main title

plt.show()
