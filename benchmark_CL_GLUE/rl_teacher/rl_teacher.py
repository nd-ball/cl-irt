from __future__ import absolute_import, division, print_function
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')
import json
import time
import argparse
import logging
import os
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler

from copy import deepcopy as cp

from utils_glue import (convert_examples_to_features, output_modes, processors)
from models import FineTunedModel, ActionPredictor
from trainer_glue import trainer

import transformers

transformers.logging.set_verbosity_error()

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'DebertaV3': (DebertaV2Config, DebertaV2ForSequenceClassification, DebertaV2Tokenizer)
}

# all_task_names = ['mrpc', 'rte', 'mnli', 'qqp', 'sst2', 'qnli']
all_task_names = ['mrpc']

class SimpleDataset(Dataset):
    def __init__(self, x1, x2, x3, x4):
        self.__iter = None
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        val = self.x1[key], self.x2[key], self.x3[key], self.x4[key]
        return val


class SimpleDataset2(Dataset):
    def __init__(self, x1, x2, x3):
        self.__iter = None
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        val = self.x1[key], self.x2[key], self.x3[key]
        return val


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, task, tokenizer, evaluate=False, held=False, test=False):
    # Load dataset
    dataset = load_dataset("glue", task)

    # Define tokenization function
    def tokenize_function(examples):
        if task in ["mnli"]:
            return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True,
                             max_length=args.max_seq_length)
        elif task in ["mrpc", "rte"]:
            return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True,
                             max_length=args.max_seq_length)
        elif task in ["qnli"]:
            return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True,
                             max_length=args.max_seq_length)
        elif task in ["qqp"]:
            return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True,
                             max_length=args.max_seq_length)
        elif task in ["sst2"]:
            return tokenizer(examples["sentence"], padding="max_length", truncation=True,
                             max_length=args.max_seq_length)

    # Select appropriate split
    if task == "mnli":
        if evaluate:
            if test:
                split = "test_matched"
            elif held:
                split = "validation_mismatched"
            else:
                split = "validation_matched"
        else:
            split = "train"
    else:
        if evaluate:
            if test:
                split = "test" if "test" in dataset.keys() else "validation"
            elif held:
                split = "validation"
            else:
                split = "validation"
        else:
            split = "train"

    # Check if the split exists
    if split not in dataset.keys():
        logger.warning(f"Split {split} not found in dataset. Available splits: {dataset.keys()}")
        return None

    logger.info(f"Loading split: {split}")

    # Tokenize and prepare dataset
    tokenized_dataset = dataset[split].map(tokenize_function, batched=True, remove_columns=dataset[split].column_names)

    # Add labels column if it exists in the original dataset
    if "label" in dataset[split].features:
        tokenized_dataset = tokenized_dataset.add_column("labels", dataset[split]["label"])

    tokenized_dataset.set_format("torch")

    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use default batch sizes if not specified in args
    train_batch_size = getattr(args, 'train_batch_size', 4)  # default to 32 if not specified
    eval_batch_size = getattr(args, 'eval_batch_size', 4)  # default to 32 if not specified

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=train_batch_size if not evaluate else eval_batch_size,
        shuffle=not evaluate,
        collate_fn=data_collator,
        num_workers=getattr(args, 'num_workers', 2)  # default to 0 if not specified
    )

    return dataloader


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", default='rte', type=str,
                        help="Model type selected in the list: " + ", ".join(all_task_names))

    parser.add_argument("--data_dir", default='./glue_data/', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--model_type", default='DebertaV3', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--teacher_model", default="microsoft/deberta-v3-base", type=str,
                        help="DebertaV3 pre-trained model selected in the list: DebertaV3, "
                        )
    parser.add_argument("--student_model", default="microsoft/deberta-v3-base", type=str,
                        help="DebertaV3 pre-trained model selected in the list: DebertaV3-base-uncased, "
                        )
    parser.add_argument('--teacher_tf_checkpoint', default=None, type=str,
                        help='Teacher TF Checkpoint')
    parser.add_argument('--student_tf_checkpoint', default=None, type=str,
                        help='Teacher TF Checkpoint')
    parser.add_argument("--not_train_teacher", action='store_true',
                        help="Whether not to train teacher.")
    parser.add_argument('--teacher_finetuned_checkpoint', default=None, type=str,
                        help='Teacher finetuned Checkpoint')

    parser.add_argument("--nlayers", default=6, type=int,
                        help="Number of encoder layers, in case the student model is not pretrained")
    parser.add_argument("--emsize", default=768, type=int,
                        help="Hidden size of embedding layer, in case the student model is not pretrained")
    parser.add_argument("--nhid", default=768, type=int,
                        help="Hidden size of encoder, in case the student model is not pretrained")
    parser.add_argument("--pooling_method", default="cls", type=str,
                        help="Pooling method for Transformer student model")
    parser.add_argument("--encoder_version", default="post", type=str,
                        help="Encoder version for Transformer, either post or ReZero")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--log_dir", default='logs', type=str, help="The log data dir.")
    parser.add_argument("--output_dir", default='tmp/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--temperature", default=5.0, type=float,
                        help="Distillation temperature for soft target.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Train loss ratio.")
    parser.add_argument("--lambda_", default=0.5, type=float,
                        help="Meta Train loss ratio.")
    parser.add_argument("--beta", default=100.0, type=float,
                        help="Distillation loss ratio.")

    parser.add_argument("--teacher_learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--meta_learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--train_dataloader_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_dataloader_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--teacher_epochs", default=2, type=int,
                        help="Number of epochs.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--num_episodes", default=200, type=int,
                        help="Number of steps for meta update.")

    parser.add_argument("--max_grad_norm", default=1, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--use_comp_loss", action='store_true',
                        help="Use competitive loss or not.")

    parser.add_argument("--reward_function", default='binary', type=str,
                        help="Type of reward function - binary or real.")

    parser.add_argument('--seed', type=int, default=63,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    args.teacher_save_path = os.path.join("output", args.task, "pytorch_model.bin")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained('microsoft/deberta-v3-base',
                                                cache_dir=args.cache_dir,
                                                use_fast=False)

    args.n_gpu = 1

    # Set seed
    set_seed(args)

    task_names = ['mrpc', 'rte', 'mnli', 'qqp', 'sst2', 'qnli']
    now = int(datetime.now().timestamp())
    args.timestamp = now

    task_loaders = {}
    label_nums = {}

    for task_name in task_names:
        # Load dataset
        args.task = task_name
        task = args.task
        dataset = load_dataset("glue", task_name.lower())

        # Get number of labels
        if task_name.lower() == "stsb":
            num_labels = 1
        elif task_name.lower() == "mnli":
            num_labels = 3
        else:
            num_labels = len(set(dataset["train"]["label"]))

        print(f"Loaded {task_name} dataset with {num_labels} labels")

        if task_name != args.task:
            held_loader = load_and_cache_examples(args, task_name, tokenizer, evaluate=True, held=True)

            if held_loader is not None:
                task_loaders[task_name] = {
                    'held': {'loader': held_loader},
                    'num_labels': num_labels
                }
            else:
                print(f"No held-out set available for {task_name}")

        else:
            train_loader = load_and_cache_examples(args, task_name, tokenizer, evaluate=False)
            held_loader = load_and_cache_examples(args, task_name, tokenizer, evaluate=True, held=True)
            eval_loader = load_and_cache_examples(args, task_name, tokenizer, evaluate=True)
            test_loader = load_and_cache_examples(args, task_name, tokenizer, evaluate=True, test=True)

            if test_loader:
                print(f"Number of test samples - {len(test_loader.dataset)}")

            task_loaders[task_name] = {
                'train': {'loader': train_loader},
                'held': {'loader': held_loader},
                'eval': {'loader': eval_loader},
                'test': {'loader': test_loader},
                'num_labels': num_labels
            }

        label_nums[task_name.lower()] = num_labels

    start_teacher_model_time = time.time()
    t_config = config_class.from_pretrained(args.teacher_model,
                                            num_labels=label_nums[args.task.lower()], finetuning_task=args.task)
    teacher_model = FineTunedModel(task_names, label_nums, t_config, pretrained_model_name=args.teacher_model, \
                                   tf_checkpoint=args.teacher_tf_checkpoint)
    end_teacher_model_time = time.time()

    start_student_model_time = time.time()
    s_config = config_class.from_pretrained(args.teacher_model,
                                            num_labels=label_nums[args.task.lower()], finetuning_task=args.task)

    student_model = FineTunedModel(task_names, label_nums, s_config, pretrained_model_name=args.teacher_model,
                                   tf_checkpoint=args.student_tf_checkpoint)
    end_student_model_time = time.time()

    start_training_model_time = time.time()
    action_model = ActionPredictor(d_model=768, num_actions=len(task_names))


    trajectories, all_rewards = trainer(args, teacher_model, student_model, action_model, task_loaders, label_nums,
                                        task)
    end_training_model_time = time.time()
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
    print(f"Time to create teacher_model: {end_teacher_model_time - start_teacher_model_time} seconds")
    print(f"Time to create student_model: {end_student_model_time - start_student_model_time} seconds")
    print(f"Time to train model: {end_training_model_time - start_training_model_time} seconds")


if __name__ == "__main__":
    main()
