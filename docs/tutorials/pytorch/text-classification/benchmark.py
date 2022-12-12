import logging
import os
import numpy as np
import random
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.optimization import OptimizedModel
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from argparse import ArgumentParser
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    TrainingArguments,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
logger = logging.getLogger(__name__)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--data_type', default = "int8", help='data type of model')
arg_parser.add_argument('--model_name_or_path', default = "textattack/bert-base-uncased-MRPC", help = 'input model for benchmark')
args = arg_parser.parse_args()

# download the dataset.
raw_datasets = load_dataset("glue", "mrpc")
# Labels
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)

training_args = TrainingArguments(
    output_dir=args.model_name_or_path,
    do_eval=True,
    do_train=True,
    no_cuda=True,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
)
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task="mrpc",
    revision="main"
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    use_fast=True,
    revision="main"
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
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        revision="main"
    )

# Preprocessing the raw_datasets
sentence1_key, sentence2_key = task_to_keys["mrpc"]
# Padding strategy
padding = False
# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warning(
            f"Your model seems to have been trained with labels, but they don't match the dataset: "
            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}.\n"
            f"Ignoring the model labels as a result."
        )
if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
max_seq_length = min(128, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

with training_args.main_process_first(desc="dataset map pre-processing"):
    raw_datasets = raw_datasets.map(
        preprocess_function, batched=True, load_from_cache_file=False
    )

if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]

if training_args.do_eval:
    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]

# Log a few random samples from the training set:
if training_args.do_train:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# Get the metric function
metric = load_metric("glue", "mrpc")

metric_name = "eval_accuracy"

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds =  np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = None

# Initialize the Trainer
set_seed(training_args.seed)
trainer = NLPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
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
