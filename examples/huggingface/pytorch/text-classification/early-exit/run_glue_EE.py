# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification with SWEET on GLUE."""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from os import path
from typing import Optional

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from models.bert.configuration_bert import BertConfig
from models.bert.modeling_bert import BertForSequenceClassification
from models.deberta.configuration_deberta import DebertaConfig
from models.deberta.modeling_deberta import DebertaForSequenceClassification
from utils.temperature_scaling import set_temperature

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")



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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    do_calibration: Optional[bool] = field(
        default=False,
        metadata={"help": "A boolean flag indicating whether the model should be calibrated prior to evaluation"}
    )
    eval_seperate_layers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "A boolean flag indicating whether the model should not be saved after evaluation to save space"}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    exit_layers: str = field(
        default=None,
        metadata={"help": "layers of the model for which to train classifiers."},
    )
    SWEET: bool = field(
        default=True,
        metadata={"help": "A flag indicating whether to use the SWEET fine tuning method."},
    )
    gold_exit_layer: int = field(
        default=None,
        metadata={"help": "An exit layer to use during evaluation."},
    )
    exit_thresholds: str = field(
        default=1,
        metadata={"help": "Number of exit threshold for inference early exit (ranged  1/num_labels to 1"},
    )
    exit_strategy: str = field(
        default='confidence_threshold',
        metadata={"help": "A function implementing a decision rule for \"early exit\""}
    )
    exit_kwargs: str = field(
        default=None,
        metadata={"help": "arguments for the exit strategy if needed"}
    )
    deberta: bool = field(
        default=False,
        metadata={
            "help": "A flag indicating whether the pretrained model is DeBERTa (by default, the pretrained model is BERT)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    do_not_save_model: bool = field(
        default=False, metadata={"help": "erase the saved model weights."}
    )


def parse_exit_thresholds(num_labels, num_thresholds, is_lte=False):
    if num_thresholds is not None:
        if is_lte:
            lower = 0
        else:
            lower = 1 / num_labels
        return list(np.linspace(lower, 1, num_thresholds))
    else:
        return [1.0]


def parse_exit_layers(exit_layers):
    if exit_layers is not None:
        return [int(i) for i in exit_layers.split('_')]
    return [-1]


def parse_exit_kwargs(exit_kwargs):
    if exit_kwargs is not None:
        ks_and_vs = exit_kwargs.split(',')
        exit_kwargs = {}
        for k_v in ks_and_vs:
            k, v = k_v.split(':')
            exit_kwargs[k] = v
        return exit_kwargs
    return {}


def move_to_device(batch, device='cuda'):
    for key, tensor in batch.items():
        batch[key] = tensor.to(device=device)
    return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.`
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    if training_args.do_eval and training_args.per_device_eval_batch_size != 1:
        training_args.per_device_eval_batch_size = 1
        print("Setting per_device_eval_batch_size to 1 since early exit works only with batch size 1.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1


    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:

            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    exit_layers = parse_exit_layers(model_args.exit_layers)
    exit_kwargs = parse_exit_kwargs(model_args.exit_kwargs)
    exit_thresholds = parse_exit_thresholds(int(num_labels), int(model_args.exit_thresholds),
                                            model_args.exit_strategy == 'lte')

    if model_args.deberta:
        my_config = DebertaConfig
        my_model = DebertaForSequenceClassification
    else:
        my_config = BertConfig
        my_model = BertForSequenceClassification

    config = my_config.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        exit_layers=exit_layers,
        SWEET=model_args.SWEET,
        gold_exit_layer=model_args.gold_exit_layer,
        exit_thresholds=exit_thresholds,
        exit_strategy=model_args.exit_strategy,
        exit_kwargs=exit_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = my_model.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    def calculate_speedup_ratio(counts):
        logger.info(f'counts = {counts}, exit layers = {exit_layers}')
        counts_times_layers = [count * (exit_layer + 1) for count, exit_layer in zip(counts, exit_layers)]
        return sum(counts_times_layers) / (sum(counts) * (exit_layers[-1] + 1))

    def compute_multiexit_metrics(p: EvalPrediction, gold_layer=None):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        # This entire segments it a workaround aimed at not changing the trainer file. We rely on all logits in a
        # prediction to be equal as an indicator of the previous layer being the exit point for the sample
        final_predictions = np.zeros(
            (preds.shape[0], preds.shape[-1]))  # will hold the predictions of the final exit layer to give a prediction
        layer_prediction_indices = {}
        layer_counts = [0 for _ in range(preds.shape[1])]

        for i, prediction in enumerate(preds):
            layer_ind = 0
            for layer_logits in prediction[1:]:
                if np.all(layer_logits == layer_logits[0]) and gold_layer is None:
                    break
                elif gold_layer == exit_layers[layer_ind]:
                    break
                else:
                    layer_ind += 1
            layer_counts[layer_ind] += 1  # add a sample prediction count to the previous layer
            final_predictions[i] = prediction[layer_ind]  # Update the actual prediction on the i-th sample
            if layer_ind not in layer_prediction_indices:  # Update that the i-th sample was classified by the
                # (j-1)st exit layer
                layer_prediction_indices[layer_ind] = []
            layer_prediction_indices[layer_ind].append(i)
        preds = np.squeeze(final_predictions) if is_regression else np.argmax(final_predictions, axis=1)

        results = metric.compute(predictions=preds, references=p.label_ids)
        results['speedup_ratio'] = calculate_speedup_ratio(layer_counts)

        metric_name = list(results.keys())[0]
        if data_args.task_name is not None or data_args.dataset_name is not None:
            for layer_ind, exit_layer in enumerate(exit_layers):
                # for layer_ind ,sample_indices in layer_prediction_indices.items():
                #     exit_layer = exit_layers[layer_ind]
                if layer_ind not in layer_prediction_indices:
                    results[f'layer_{exit_layer}'] = {metric_name: 0, 'counts': layer_counts[layer_ind]}
                else:
                    sample_indices = layer_prediction_indices[layer_ind]
                    results[f'layer_{exit_layer}'] = metric.compute(predictions=preds[sample_indices],
                                                                    references=(p.label_ids)[sample_indices])
                    results[f'layer_{exit_layer}']['counts'] = layer_counts[layer_ind]

            return results
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    label_to_id = None
    if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Shuffle and make sure samples are identical for same seeds across experiments
    for dataset in raw_datasets.values():
        dataset.shuffle(seed=training_args.seed)
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_multiexit_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Training

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        if not model_args.do_not_save_model:
            trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Calibration
    if data_args.do_calibration:
        calibration_dataset = eval_dataset
        if model_args.exit_strategy == 'lte':
            min_lte, max_lte = None, None
            eval_dataloader = trainer.get_eval_dataloader(calibration_dataset)
            for step, batch in enumerate(eval_dataloader):
                batch = move_to_device(batch, training_args.device)
                with torch.no_grad():
                    outputs = model(**batch, output_hidden_states=True)
                if type(outputs.hidden_states) is tuple:
                    logger.info(f'hidden states is a tuple: {outputs.hidden_states}')
                lte_logits = outputs.hidden_states.cpu()
                if min_lte is None:
                    min_lte = lte_logits
                if max_lte is None:
                    max_lte = lte_logits
                min_lte = torch.minimum(min_lte, lte_logits)
                max_lte = torch.maximum(max_lte, lte_logits)
            model.set_min_max_lte(min_lte, max_lte)
        else:
            temperature_file_path = path.join(training_args.output_dir, 'scaling_temperatures.txt')
            if path.exists(temperature_file_path):
                with open(temperature_file_path, 'r') as temp_file:
                    temps = temp_file.readlines()
            else:
                temps = set_temperature(model, len(model.exit_layers), calibration_dataset)
                with open(temperature_file_path, 'a') as temp_file:
                    for t in temps:
                        temp_file.write(str(t) + '\n')
            logger.info(f'Calibrated temperatures: {temps}')
            model.set_scaling_temperatures(temps)
    model = model.to('cpu')
    # Evaluation

    if training_args.do_eval:
        # training_args.output_dir = base_output_dir
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_multiexit_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]

        # If you want to automatically handle mismatched when handling mnli, uncomment this segment (and debug):
        # if data_args.task_name == "mnli":
        #     tasks.append("mnli-mm")
        #     mm_dataset = raw_datasets["validation_mismatched"]
        #     if data_args.max_eval_samples is not None:
        #         mm_dataset = mm_dataset.select(range(data_args.max_eval_samples))
        #     eval_datasets.append(mm_dataset)
        total_metrics = {}
        for eval_dataset, task in zip(eval_datasets, tasks):
            for thr in exit_thresholds:
                model.set_exit_strategy(model_args.exit_strategy, threshold=thr)
                metrics = trainer.evaluate(eval_dataset=eval_dataset)
                max_eval_samples = (
                    data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset))
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                if task == "mnli-mm":
                    metrics = {k + "_mm": v for k, v in metrics.items()}

                # If you want to automatically handle mismatched when handling mnli, uncomment this segment (and debug):
                # if task is not None and "mnli" in task:
                #     combined.update(metrics)
                # total_metrics[thr] = combined if task is not None and "mnli" in task else metrics
                total_metrics[thr] = metrics
                metrics_for_logging = {k: v for k, v in metrics.items() if 'layer' not in k}
                trainer.log_metrics(f"eval_for_thr {thr}", metrics_for_logging)
            trainer.save_metrics(f"threshold", total_metrics, combined=False)
        model.to(training_args.device)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
