#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional, Union
import numpy as np

import datasets
import tensorflow as tf
from datasets import load_dataset

import time

import transformers
from transformers import (
    CONFIG_NAME,
    TF2_WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
    DefaultDataCollator,
    HfArgumentParser,
    PushToHubCallback,
    TFAutoModelForMultipleChoice,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)


# region Helper classes and functions


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="tf",
        )

        # Un-flatten
        batch = {k: tf.reshape(v, (batch_size, num_choices, -1)) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = tf.convert_to_tensor(labels, dtype=tf.int64)
        return batch


# endregion

# region Arguments
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

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
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

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."



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
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args, distributed_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, optim_args, distributed_args = parser.parse_args_into_dataclasses()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # endregion

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

    # region Set the multinode environment, the strategy and paths
    strategy = None
    worker_list = None
    if distributed_args.worker is not None:
        logger.info("distributed environment initialization...")

        worker_list = distributed_args.worker.split(",")

        from intel_extension_for_transformers.transformers.utils.utility_tf import distributed_init
        distributed_init(worker_list, "worker", distributed_args.task_index)

        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        from intel_extension_for_transformers.transformers.utils.utility_tf import get_filepath
        training_args.output_dir = get_filepath(training_args.output_dir, strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
    else:
        strategy = training_args.strategy
    #endregion

    # region Checkpoints
    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        if (output_dir / CONFIG_NAME).is_file() and (output_dir / TF2_WEIGHTS_NAME).is_file():
            checkpoint = output_dir
            logger.info(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )
    # endregion

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # region Load datasets
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Downloading and loading the swag dataset from the hub.
        raw_datasets = load_dataset(
            "swag",
            "regular",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    # endregion

    # region Load model config and tokenizer
    if checkpoint is not None:
        config_path = training_args.output_dir
    elif model_args.config_name:
        config_path = model_args.config_name
    else:
        config_path = model_args.model_name_or_path

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        config_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _commit_hash="main",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        _commit_hash="main",
    )
    # endregion

    # region Dataset preprocessing
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

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
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=max_seq_length)
        # Un-flatten
        data = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return data


    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )


    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.pad_to_max_length:
        data_collator = DefaultDataCollator(return_tensors="tf")
    else:
        # custom class defined above, as HF has no data collator for multiple choice
        data_collator = DataCollatorForMultipleChoice(tokenizer)
    # endregion

    with strategy.scope():
        # region Build model
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint
        model = TFAutoModelForMultipleChoice.from_pretrained(
            model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = training_args.per_device_train_batch_size * num_replicas
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas

        num_train_steps = (len(train_dataset) // total_train_batch_size) * int(training_args.num_train_epochs)
        if training_args.warmup_steps > 0:
            num_warmup_steps = training_args.warmup_steps
        elif training_args.warmup_ratio > 0:
            num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
        else:
            num_warmup_steps = 0
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
            adam_global_clipnorm=training_args.max_grad_norm,
        )


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
            shuffle=True,
            batch_size=total_train_batch_size,
            collate_fn=data_collator,
        ).with_options(dataset_options)

        tf_eval_dataset = model.prepare_tf_dataset(
            eval_dataset,
            shuffle=False,
            batch_size=total_eval_batch_size,
            collate_fn=data_collator,
            drop_remainder=True,
        ).with_options(dataset_options)

        model.compile(optimizer=optimizer, metrics=["accuracy"], jit_compile=training_args.xla)
        # endregion

    def compute_metrics(preds, labels):
        predictions = preds["logits"]
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == labels).astype(np.float32).mean().item()}

    # region tuning
    if optim_args.tune:
        from intel_extension_for_transformers.transformers import metrics, objectives, QuantizationConfig, TFOptimization
        optimization = TFOptimization(
            model=model,
            args=training_args,
            train_dataset=tf_train_dataset,
            eval_dataset=tf_eval_dataset,
            compute_metrics=compute_metrics,
            task_type=strategy.cluster_resolver.task_type if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy) else None,
            task_id=strategy.cluster_resolver.task_id if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy) else None,
        )

        # use customized eval function
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
    # endregion

    # region Training
    eval_metrics = None
    if training_args.do_train:
        history = model.fit(
            tf_train_dataset,
            validation_data=tf_eval_dataset,
            epochs=int(training_args.num_train_epochs),
        )
        model.save("finetuned_model")
        eval_metrics = {key: val[-1] for key, val in history.history.items()}
        # endregion

    # region Evaluation
    if training_args.do_eval:
        num_examples = sum(1 for _ in (
            tf_eval_dataset.unbatch() if hasattr(tf_eval_dataset, "unbatch") else tf_eval_dataset))

        if optim_args.int8:
            model = tf.saved_model.load(training_args.output_dir)
        else:
            from intel_extension_for_transformers.transformers.utils.utility_tf import keras2SavedModel
            model = keras2SavedModel(model)

        logger.info(f"***** Running Evaluation *****")
        logger.info(f"  Num examples in dataset = {num_examples}")
        logger.info(f"  Batch size = {training_args.per_device_eval_batch_size}")

        preds: np.ndarray = None
        label_ids: np.ndarray = None
        infer = model.signatures["serving_default"]

        if optim_args.accuracy_only:
            iterations = 1
            warmup = 0
        else:
            iterations = 10
            warmup = 5
        latency_list = []

        for idx in range(iterations):
            iteration_time = 0
            for i, (inputs, labels) in enumerate(tf_eval_dataset):
                for name in inputs:
                    inputs[name] = tf.constant(inputs[name].numpy(), dtype=infer.inputs[0].dtype)

                start = time.time()
                results = infer(**inputs)
                iteration_time += time.time() - start
                if idx == 0:    # only accumulate once all the preds and labels
                    if preds is None:
                        preds = results["Identity"].numpy()
                    else:
                        preds = np.append(preds, results["Identity"].numpy(), axis=0)
                    if label_ids is None:
                        label_ids = labels[0].numpy() if isinstance(
                            labels, list) else labels.numpy()
                    else:
                        label_ids = np.append(
                            label_ids,
                            labels[0].numpy() if isinstance(labels, list) else labels.numpy(),
                            axis=0)
            latency_list.append(iteration_time)
            logger.info("Iteration {} time: {} sec".format(idx, iteration_time))

        test_predictions = {"logits": preds}
        eval_metrics = compute_metrics(test_predictions, label_ids)
        logger.info("\nEvaluation result: ")
        logger.info("Accuracy: {}".format(eval_metrics["accuracy"]))

        average_iteration_time = np.array(latency_list[warmup:]).mean()
        logger.info(
            "Throughput: {} samples/sec".format(
                num_examples / average_iteration_time)
        )

if __name__ == "__main__":
    main()