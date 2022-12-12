import logging
import os
import numpy as np
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.optimization import OptimizedModel
from intel_extension_for_transformers.optimization.trainer import  NLPSeq2SeqTrainer
from argparse import ArgumentParser
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--data_type', default = "int8", help='data type of model')
arg_parser.add_argument('--model_name_or_path', default = "t5-small", help = 'input model for benchmark')
args = arg_parser.parse_args()

raw_datasets = load_dataset("wmt16", "ro-en")
training_args = Seq2SeqTrainingArguments(
    output_dir="./saved_results_dynamic",
    do_eval=True,
    do_train=True,
    no_cuda=True,
    overwrite_output_dir=True,
    per_device_eval_batch_size=8,
    predict_with_generate=True
)
config = AutoConfig.from_pretrained(args.model_name_or_path, revision="main")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision="main", use_fast=True)
prefix = ""

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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "t5-small",
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        revision="main"
    )
    model.resize_token_embeddings(len(tokenizer))

# We need to tokenize inputs and targets.
column_names = raw_datasets["train"].column_names

# Get the language codes for input/target.
source_lang = "en"
target_lang = "ro"

# Temporarily set max_target_length for training.
max_target_length = 128
padding = False

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# define train dataset
train_dataset = raw_datasets["train"]
with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

# define eval dataset
eval_dataset = raw_datasets["validation"]
max_eval_samples = min(len(eval_dataset), 400)
eval_dataset = eval_dataset.select(range(max_eval_samples))
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on validation dataset",
    )

# Data collator
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

# Metric
metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

metric_name = "eval_bleu"
max_length = 128
num_beams = None
# Initialize the Trainer
set_seed(training_args.seed)
trainer = NLPSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)

results = trainer.evaluate(max_length=max_length, num_beams=num_beams)
bert_task_acc_keys = ['eval_loss', 'eval_f1', 'eval_accuracy', 'eval_matthews_correlation',
                              'eval_pearson', 'eval_mcc', 'eval_spearmanr']

throughput = results.get("eval_samples_per_second")
eval_loss = results["eval_loss"]
print('Batch size = {}'.format(training_args.per_device_eval_batch_size))
print("Finally Eval eval_loss Accuracy: {}".format(eval_loss))
print("Latency: {:.3f} ms".format(1000 / throughput))
print("Throughput: {} samples/sec".format(throughput))
