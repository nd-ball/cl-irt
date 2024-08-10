import os
import csv
import torch
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = str(text_a)
        self.text_b = str(text_b) if text_b is not None else None
        self.label = str(label) if label is not None else None

class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def read_tsv(input_file):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, dtype=str).values.tolist()

class DataProcessor:
    def get_train_examples(self, data_dir):
        return self._create_examples(self.read_tsv(os.path.join(data_dir, self.task_name, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.read_tsv(os.path.join(data_dir, self.task_name, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self.read_tsv(os.path.join(data_dir, self.task_name, "test.tsv")), "test")

    def get_held_examples(self, data_dir):
        return self._create_examples(self.read_tsv(os.path.join(data_dir, self.task_name, "held.tsv")), "held")

    @staticmethod
    def read_tsv(input_file):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return pd.read_csv(f, sep="\t", quoting=csv.QUOTE_NONE, dtype=str).values.tolist()

    def _create_examples(self, lines, set_type):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    def get_held_examples(self, data_dir):
        file_path = os.path.join(data_dir, self.task_name, "held.tsv")
        if os.path.exists(file_path):
            return self._create_examples(self.read_tsv(file_path), "held")
        else:
            logger.warning(f"held.tsv not found in {os.path.join(data_dir, self.task_name)}. Returning empty list.")
            return []


class MrpcProcessor(DataProcessor):
    def __init__(self):
        self.task_name = "mrpc"

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            try:
                if len(line) >= 5:  # Full format
                    text_a = line[3]
                    text_b = line[4]
                    label = line[0] if set_type != "test" else None
                elif len(line) == 3:  # Minimal format (assuming id, text_a, text_b)
                    text_a = line[1]
                    text_b = line[2]
                    label = None
                else:
                    logger.warning(f"Skipping line {i} due to unexpected format: {line}")
                    continue

                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            except IndexError:
                logger.warning(f"Skipping line {i} due to IndexError: {line}")
                continue
        return examples

# Add other processors (RteProcessor, MnliProcessor, etc.) here...
class RteProcessor(DataProcessor):
    def __init__(self):
        self.task_name = "rte"
    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[1]
            text_b = line[2]
            label = line[3] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliProcessor(DataProcessor):
    def __init__(self):
        self.task_name = "mnli"
    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[8]
            text_b = line[9]
            label = line[-1] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class QqpProcessor(DataProcessor):
    def __init__(self):
        self.task_name = "qqp"
    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = line[5] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Sst2Processor(DataProcessor):
    def __init__(self):
        self.task_name = "sst2"
    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = line[1] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class QnliProcessor(DataProcessor):
    def __init__(self):
        self.task_name = "qnli"
    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[1]
            text_b = line[2]
            label = line[3] if set_type != "test" else None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b) if example.text_b else None

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label is not None:
            if example.label in label_map:
                label_id = label_map[example.label]
            else:
                logger.warning(f"Label '{example.label}' not found in label_map. Using -1 as label_id.")
                label_id = -1
        else:
            label_id = -1

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      segment_ids=segment_ids, label_id=label_id))
    return features
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return accuracy_score(labels, preds)

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(labels, preds, average='weighted')
    return {"acc": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["mrpc", "qqp"]:
        return acc_and_f1(preds, labels)
    elif task_name in ["mnli", "rte", "sst2", "qnli"]:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "mrpc": MrpcProcessor,
    "rte": RteProcessor,
    "mnli": MnliProcessor,
    "qqp": QqpProcessor,
    "sst2": Sst2Processor,
    "qnli": QnliProcessor,
}

output_modes = {
    "mrpc": "classification",
    "rte": "classification",
    "mnli": "classification",
    "qqp": "classification",
    "sst2": "classification",
    "qnli": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "mrpc": 2,
    "rte": 2,
    "mnli": 3,
    "qqp": 2,
    "sst2": 2,
    "qnli": 2,
}