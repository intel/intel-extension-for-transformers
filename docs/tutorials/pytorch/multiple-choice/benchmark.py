import logging
import os
import numpy as np
from datasets import load_dataset, load_metric
from itertools import chain
from intel_extension_for_transformers.optimization import metrics, OptimizedModel
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from argparse import ArgumentParser
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--data_type', default = "int8", help='data type of model')
arg_parser.add_argument('--model_name_or_path', default = "ehdwns1516/bert-base-uncased_SWAG", help = 'input model for benchmark')
args = arg_parser.parse_args()

dataset_name="swag"
dataset_config_name="regular"

training_args = TrainingArguments(
    output_dir=args.model_name_or_path,
    do_eval=True,
    do_train=True,
    no_cuda=True,
    overwrite_output_dir=True,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8
)

raw_datasets = load_dataset(dataset_name, dataset_config_name)
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        config=config,
        revision="main",
        use_auth_token=None,
    )

ending_names = [f"ending{i}" for i in range(4)]
context_name = "sent1"
question_header_name = "sent2"

# First we tokenize all the texts.
max_seq_length = tokenizer.model_max_length
if max_seq_length >1024:
    max_seq_length = 1024

# preprocessing the datasets
def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples[context_name]]
    question_headers = examples[question_header_name]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length"
    )
    # Un-flatten
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(preprocess_function, batched=True)
if training_args.do_eval:
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.select(range(1000))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=False
        )

# Data collator
data_collator = default_data_collator

# Metric
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

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
