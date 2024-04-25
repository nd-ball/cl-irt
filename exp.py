from datasets import load_dataset
import torch
# List of GLUE benchmark tasks you want to load
GLUE_TASKS = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]

# Check if GPU is available
GPU_avail = torch.cuda.is_available()
print("GPU_CUDA is available:", GPU_avail)

# Define batch size and max length
batch_size = 64
max_length = 512

# List of model checkpoints to consider
models = ["gpt2"]

for model_checkpoint in models:
    for task in GLUE_TASKS:
        print("Task:", task)
        dataset = load_dataset("glue", task)
        # Further processing or analysis with the loaded dataset

