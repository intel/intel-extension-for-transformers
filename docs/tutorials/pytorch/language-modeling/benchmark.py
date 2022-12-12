import logging
import os
from datasets import load_dataset, load_metric
from itertools import chain
from intel_extension_for_transformers.optimization import metrics, OptimizedModel
from intel_extension_for_transformers.optimization.trainer import NLPTrainer
from argparse import ArgumentParser
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--data_type', default = "int8", help='data type of model')
arg_parser.add_argument('--model_name_or_path', default = "bert-base-uncased", help = 'input model for benchmark')
args = arg_parser.parse_args()

dataset_name="wikitext"
dataset_config_name="wikitext-2-raw-v1"
training_args = TrainingArguments(
    output_dir=args.mpdel_name_or_path,
    do_eval=True,
    do_train=True,
    no_cuda=True,
    per_device_eval_batch_size=1,
    overwrite_output_dir=True
)

raw_datasets = load_dataset(dataset_name, dataset_config_name)
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# Set seed before initializing model.
set_seed(training_args.seed)

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
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        revision="main",
        use_auth_token=None,
    )
    model.resize_token_embeddings(len(tokenizer))

# First we tokenize all the texts.
if training_args.do_train:
    column_names = raw_datasets["train"].column_names
else:
    column_names = raw_datasets["validation"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

max_seq_length = tokenizer.model_max_length


def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)


column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on every text in dataset",
    )


# Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# might be slower to preprocess.

with training_args.main_process_first(desc="grouping texts together"):
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

if training_args.do_train:
    if "train" not in tokenized_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = tokenized_datasets["train"]

if training_args.do_eval:
    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = tokenized_datasets["validation"]


    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


    metric = load_metric("accuracy")


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

# Data collator will take care of randomly masking the tokens.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
    pad_to_multiple_of=None,
)

# Initialize the Trainer
set_seed(training_args.seed)
trainer = NLPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
    if training_args.do_eval and not is_torch_tpu_available()
    else None,
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
