import logging
import os
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
from intel_extension_for_transformers.optimization import OptimizedModel
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from argparse import ArgumentParser
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    TrainingArguments,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--data_type', default = "int8", help='data type of model')
arg_parser.add_argument('--model_name_or_path', default = "elastic/distilbert-base-uncased-finetuned-conll03-english", help = 'input model for benchmark')
args = arg_parser.parse_args()

# download the dataset.
raw_datasets = load_dataset("conll2003")
training_args = TrainingArguments(
    output_dir=args.model_name_or_path,
    do_eval=True,
    do_train=True,
    no_cuda=True,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)
column_names = raw_datasets["train"].column_names
features = raw_datasets["train"].features
text_column_name = "tokens"
label_column_name = "ner_tags"

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

# If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
# Otherwise, we have to get the list of labels manually.
labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
if labels_are_int:
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}

num_labels = len(label_list)

# download model & vocab.
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task="ner",
    revision="main",
)

tokenizer_name_or_path = args.model_name_or_path
if config.model_type in {"gpt2", "roberta"}:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        revision="main",
        add_prefix_space=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        revision="main",
    )

## start with int8 benchmarking
if args.data_type == "int8":
    # Load the model obtained after Intel Neural Compressor (INC) quantization
    model = OptimizedModel.from_pretrained(
          args.model_name_or_path,
          from_tf=bool(".ckpt" in args.model_name_or_path),
          config=config,
          revision="main",
          use_auth_token=None,
    )
else:
    ## original fp32 model benchmarking
    # Load the model obtained after Intel Neural Compressor (INC) quantization
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        revision="main"
    )
# Model has labels -> use them.
if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
        # Reorganize `label_list` to match the ordering of the model.
        if labels_are_int:
            label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
            label_list = [model.config.id2label[i] for i in range(num_labels)]
        else:
            label_list = [model.config.id2label[i] for i in range(num_labels)]
            label_to_id = {l: i for i, l in enumerate(label_list)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels: {list(sorted(label_list))}."
            "\nIgnoring the model labels as a result.",
        )

# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}

# Map that sends B-Xxx label to its I-Xxx counterpart
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)

# Padding strategy
padding = "max_length"

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# train dataset
train_dataset = raw_datasets["train"]
with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

# evaluation dataset
eval_dataset = raw_datasets["validation"]
eval_dataset = eval_dataset.select(range(1000))
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on validation dataset",
    )

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics
metric = load_metric("seqeval")
metric_name = "eval_f1"

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Initialize the Trainer
set_seed(training_args.seed)
trainer = NLPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

results = trainer.evaluate()
bert_task_acc_keys = ['eval_loss', 'eval_f1', 'eval_accuracy', 'eval_matthews_correlation',
                              'eval_pearson', 'eval_mcc', 'eval_spearmanr']

throughput = results.get("eval_samples_per_second")
eval_loss = results["eval_loss"]
print('Batch size = {}'.format(training_args.per_device_eval_batch_size))
print("Finally Eval eval_loss Accuracy: {}".format(eval_loss))
print("Latency: {:.3f} ms".format(1000 / throughput))
print("Throughput: {} samples/sec".format(throughput))
