#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import logging
import sys
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from datasets import ClassLabel, load_dataset, load_metric
import numpy as np
import tensorflow as tf

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    TFAutoModelForTokenClassification,
    TFTrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/tensorflow/token-classification/requirements.txt")

class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


# region Command-line arguments
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
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(default=256, metadata={"help": "Max length (in tokens) for truncation/padding"})
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
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
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    tune: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    quantization_approach: Optional[str] = field(
        default="PostTrainingStatic",
        metadata={"help": "Quantization approach. Supported approach are PostTrainingStatic, "
                  "PostTrainingDynamic and QuantizationAwareTraining."},
    )
    metric_name: Optional[str] = field(
        default=None,
        metadata={"help": "Metric used for the tuning strategy."},
    )
    is_relative: Optional[bool] = field(
        default=True,
        metadata={"help": "Metric tolerance model, expected to be relative or absolute."},
    )
    perf_tol: Optional[float] = field(
        default=0.01,
        metadata={"help": "Performance tolerance when optimizing the model."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "run benchmark."})
    int8: bool = field(
        default=False,
        metadata={"help":"Whether to use the quantized int8 model."})
    accuracy_only: bool = field(
        default=False,
        metadata={"help":"Whether to only test accuracy for model tuned by Neural Compressor."})

@dataclass
class DistributedArguments:
    """
    Arguments setting the distributed multinode environment
    """

    worker: str = field(
        default=None,
        metadata={"help": "List of node ip addresses in a string, and there should not be space between addresses."},
    )
    task_index: int = field(
        default=0,
        metadata={"help": "Worker index, and 0 represents the chief worker while other workers are set as 1,2,3..."},
    )

# endregion


def main():
    # region Argument Parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments, OptimizationArguments, DistributedArguments))
    model_args, data_args, training_args, optim_args, distributed_args = parser.parse_args_into_dataclasses()
    # endregion

    # region Setup logging
    # we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)
    # endregion

    # region Set the multinode environment, the strategy and paths
    strategy = None
    worker_list = None
    if distributed_args.worker is not None:
        logger.info("distributed environment initialization...")

        worker_list = distributed_args.worker.split(",")

        from intel_extension_for_transformers.optimization.utils.utility_tf import distributed_init
        distributed_init(worker_list, "worker", distributed_args.task_index)

        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        from intel_extension_for_transformers.optimization.utils.utility_tf import get_filepath
        training_args.output_dir = get_filepath(training_args.output_dir, strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
    else:
        strategy = training_args.strategy
    #endregion

    # region Loading datasets
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=True if model_args.use_auth_token else None,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            use_auth_token=True if model_args.use_auth_token else None,
            cache_dir=model_args.cache_dir,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    # endregion

    # region Load config and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _commit_hash="main",
    )

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=model_args.cache_dir, use_fast=True,
         add_prefix_space=True, _commit_hash="main",)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=model_args.cache_dir, use_fast=True,
         _commit_hash="main",)
    # endregion
    
    # region Preprocessing the raw datasets
    # First we tokenize all the texts.
    # should always use padding because the current ptq does not use tf > 2.8
    # so no RaggedTensor is supported
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=data_args.max_length,
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
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # endregion

    # Metrics
    metric = load_metric("seqeval")

    def get_labels(y_pred, y_true):
        # Transform predictions and references tensos to numpy arrays

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics(predictions, labels):
        predictions = predictions["logits"]
        predictions = np.argmax(predictions, axis=-1)

        attention_mask = eval_dataset.with_format("tf")["attention_mask"]
        labels[attention_mask == 0] = -100

        # Remove ignored index (special tokens)
        preds, refs = get_labels(predictions, labels)

        metric.add_batch(
            predictions=preds,
            references=refs,
        )
        results = metric.compute()

        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # endregion

    with strategy.scope():
        # region Initialize model
        if model_args.model_name_or_path:
            model = TFAutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            logger.info("Training new model from scratch")
            model = TFAutoModelForTokenClassification.from_config(config)

        model.resize_token_embeddings(len(tokenizer))
        # endregion

        # region Create TF datasets

        # We need the DataCollatorForTokenClassification here, as we need to correctly pad labels as
        # well as inputs.
        collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
        total_train_batch_size = training_args.per_device_train_batch_size * (len(worker_list) if worker_list is not None else 1)

        dataset_options = tf.data.Options()
        dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        # model.prepare_tf_dataset() wraps a Hugging Face dataset in a tf.data.Dataset which is ready to use in
        # training. This is the recommended way to use a Hugging Face dataset when training with Keras. You can also
        # use the lower-level dataset.to_tf_dataset() method, but you will have to specify things like column names
        # yourself if you use this method, whereas they are automatically inferred from the model input names when
        # using model.prepare_tf_dataset()
        # For more info see the docs:
        # https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset
        # https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset

        tf_train_dataset = model.prepare_tf_dataset(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=total_train_batch_size,
            shuffle=True,
        ).with_options(dataset_options)
        total_eval_batch_size = training_args.per_device_eval_batch_size * (len(worker_list) if worker_list is not None else 1)
        tf_eval_dataset = model.prepare_tf_dataset(
            eval_dataset,
            collate_fn=collate_fn,
            batch_size=total_eval_batch_size,
            shuffle=False,
        ).with_options(dataset_options)

        # endregion

        # region Optimizer, loss and compilation
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_args.learning_rate,
            beta_1=training_args.adam_beta1,
            beta_2=training_args.adam_beta2,
            epsilon=training_args.adam_epsilon,
            clipnorm=training_args.max_grad_norm,
        )

        model.compile(optimizer=optimizer, jit_compile=training_args.xla)
        # endregion

    if optim_args.tune:
        from intel_extension_for_transformers.optimization import metrics, objectives, QuantizationConfig, TFOptimization
        optimization = TFOptimization(
            model=model,
            args=training_args,
            train_dataset=tf_train_dataset,
            eval_dataset=tf_eval_dataset,
            compute_metrics=compute_metrics,
            task_type=strategy.cluster_resolver.task_type if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy) else None,
            task_id=strategy.cluster_resolver.task_id if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy) else None,
        )
        tune_metric = metrics.Metric(
            name="accuracy", greater_is_better=True, is_relative=True, criterion=optim_args.perf_tol,
        )
        quantization_config = QuantizationConfig(
            framework="tensorflow",
            approach="POSTTRAININGSTATIC",
            metrics=[tune_metric],
            objectives=[objectives.performance]
        )
        quantized_model = optimization.quantize(quant_config=quantization_config)
        exit(0)

    # region Training
    if training_args.do_train:
        callbacks = [SavePretrainedCallback(output_dir=training_args.output_dir)]
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {total_train_batch_size}")
        # Only show the progress bar once on each machine.

        model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
            callbacks=callbacks,
        )
    # endregion

    # region Evaluation
    if training_args.do_eval:
        # We normally do validation as part of the Keras fit loop, but we run it independently
        # if there was no fit() step (because we didn't train the model) or if the task is MNLI,
        # because MNLI has a separate validation-mismatched validation set
        logger.info("*** Evaluate ***")

        tasks = [data_args.task_name]
        tf_datasets = [tf_eval_dataset]
        raw_datasets = [processed_raw_datasets["validation"]]

        num_examples = 0

        if optim_args.int8:
            model = tf.saved_model.load(training_args.output_dir)
        else:
            from intel_extension_for_transformers.optimization.utils.utility_tf import keras2SavedModel
            model = keras2SavedModel(model)

        for raw_dataset, tf_dataset, task in zip(raw_datasets, tf_datasets, tasks):
            num_examples += sum(
                1 for _ in (tf_dataset.unbatch()
                            if hasattr(tf_dataset, "unbatch") else tf_dataset
                            )
            )

            preds: np.ndarray = None
            label_ids: np.ndarray = None
            infer = model.signatures[list(model.signatures.keys())[0]]

            if optim_args.accuracy_only:
                iterations = 1
                warmup = 0
            else:
                iterations = 10
                warmup = 5
            latency_list = []

            for idx in range(iterations):
                iteration_time = 0
                for i, (inputs, labels) in enumerate(tf_dataset):
                    for name in inputs:
                        inputs[name] = tf.constant(inputs[name].numpy(), dtype=infer.inputs[0].dtype)
                    start = time.time()
                    results = infer(**inputs)
                    iteration_time += time.time() - start
                    if idx == 0:    # only accumulate once all the preds and labels
                        for val in results:
                            if preds is None:
                                preds = results[val].numpy()
                            else:
                                preds = np.append(preds, results[val].numpy(), axis=0)
                        if label_ids is None:
                            label_ids = labels.numpy()
                        else:
                            label_ids = np.append(label_ids, labels.numpy(), axis=0)

                latency_list.append(iteration_time)
                logger.info("Iteration {} time: {} sec".format(idx, iteration_time))
            eval_metrics = compute_metrics({"logits": preds}, label_ids)
            logger.info("\nEvaluation result: ")
            logger.info("metric ({}) Accuracy: {}".format(task, eval_metrics["accuracy"]))

            average_iteration_time = np.array(latency_list[warmup:]).mean()
            logger.info(
                "Throughput: {} samples/sec".format(
                    num_examples / average_iteration_time)
            )
    # endregion

if __name__ == "__main__":
    main()