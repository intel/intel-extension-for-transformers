import logging
import os
import numpy as np
import nltk 
from datasets import load_dataset, load_metric
from intel_extension_for_transformers.optimization import metrics, OptimizedModel
from intel_extension_for_transformers.optimization.trainer import NLPSeq2SeqTrainer
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
arg_parser.add_argument('--model_name_or_path', default = "lvwerra/pegasus-samsum", help = 'input model for benchmark')
args = arg_parser.parse_args()

dataset_name="samsum"
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}
training_args = Seq2SeqTrainingArguments(
    output_dir=args.model_name_or_path,
    do_eval=True,
    do_train=True,
    no_cuda=True,
    predict_with_generate=True,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

raw_datasets = load_dataset(dataset_name)
config = AutoConfig.from_pretrained(args.model_name_or_path, revision="main")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, revision="main")

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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        revision="main"
    )
    model.resize_token_embeddings(len(tokenizer))

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

if (
    hasattr(model.config, "max_position_embeddings")
    and model.config.max_position_embeddings < 1024
):
    model.resize_position_embeddings(1024)

prefix = ""
# preprocessing dataset

# Preprocessing the datasets.
# We need to tokenize inputs and targets.
if training_args.do_train:
    column_names = raw_datasets["train"].column_names
elif training_args.do_eval:
    column_names = raw_datasets["validation"].column_names
elif training_args.do_predict:
    column_names = raw_datasets["test"].column_names
else:
    logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")


# Get the column names for input/target.
dataset_columns = summarization_name_mapping.get(dataset_name, None)
text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]

# Temporarily set max_target_length for training.
max_target_length = 128
padding = False

def preprocess_function(examples):
    # remove pairs where at least one record is None

    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] is not None and examples[summary_column][i] is not None:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if training_args.do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    max_train_samples = min(len(train_dataset), 10000)
    train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )

if training_args.do_eval:
    max_target_length = 128
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    max_eval_samples = min(len(eval_dataset), 500)
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
metric = load_metric("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

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

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

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
max_length = (
    training_args.generation_max_length
    if training_args.generation_max_length is not None
    else 128
)
num_beams = training_args.generation_num_beams
trainer.max_length = max_length
trainer.num_beams = num_beams

results = trainer.evaluate(max_length=max_length, num_beams=num_beams)
bert_task_acc_keys = ['eval_loss', 'eval_f1', 'eval_accuracy', 'eval_matthews_correlation',
                              'eval_pearson', 'eval_mcc', 'eval_spearmanr']

throughput = results.get("eval_samples_per_second")
eval_loss = results["eval_loss"]
print('Batch size = {}'.format(training_args.per_device_eval_batch_size))
print("Finally Eval eval_loss Accuracy: {}".format(eval_loss))
print("Latency: {:.3f} ms".format(1000 / throughput))
print("Throughput: {} samples/sec".format(throughput))
