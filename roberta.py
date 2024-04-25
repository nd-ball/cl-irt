from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer,EvalPrediction
import accelerate
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

GLUE_TASKS = ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]
datasets = {task: load_dataset("glue", task) for task in GLUE_TASKS}


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
max_length=128
def preprocess_function(examples, task):
    if task in ["mnli"]:
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", truncation=True, max_length=max_length)
    elif task in ["mrpc", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         padding="max_length", truncation=True, max_length=max_length)
    elif task in ["qnli"]:
        return tokenizer(examples["question"], examples["sentence"],
                         padding="max_length", truncation=True, max_length=max_length)
    elif task in ["qqp"]:
        return tokenizer(examples["question1"], examples["question2"],
                         padding="max_length", truncation=True, max_length=max_length)

    elif task in ["sst2"]:
        return tokenizer(examples["sentence"],
                         padding="max_length", truncation=True, max_length=max_length)

preprocessed_datasets = {task: datasets[task].map(lambda examples: preprocess_function(examples, task), batched=True) for task in GLUE_TASKS}

models = {}
for task in GLUE_TASKS:
    num_labels = 3 if task == "mnli" else 1 if task in ["stsb"] else 2  # Adjust based on the task
    models[task] = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=num_labels)




# def compute_metrics(eval_pred):
#     print(eval_pred)
#     logits, labels = eval_pred
#     if len(logits) == 2:  # This check is for models that output logits in a different format, e.g., (logits, some_other_output)
#         logits = logits[0]
#     predictions = np.argmax(logits, axis=-1)
#     return {"accuracy": accuracy_score(labels, predictions)}


metric = evaluate.load("glue", task)
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

# def compute_metrics(task):
#     # Define this function based on the metrics required for each task, similar to the previous example
#     pass

for task, model in models.items():
    training_args = TrainingArguments(
        output_dir=f'./results/{task}',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_dir=f'./logs/{task}',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=preprocessed_datasets[task]['train'],
        train_dataset=preprocessed_datasets[task]['validation_matched' if task == "mnli" else 'validation'],
        eval_dataset=preprocessed_datasets[task]['train'],
        # eval_dataset=preprocessed_datasets[task]['validation_matched' if task == "mnli" else 'validation'],
        compute_metrics=compute_metrics  # You should define compute_metrics for each task
    )

    trainer.train()
    trainer.evaluate()