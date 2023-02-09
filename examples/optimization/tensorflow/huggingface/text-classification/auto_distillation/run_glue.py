#!/usr/bin/env python
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
import numpy as np
import tensorflow as tf
import time
import transformers
from dataclasses import dataclass, field
from typing import List, Optional

from datasets import load_dataset, load_metric

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    DefaultDataCollator,
    HfArgumentParser,
    PretrainedConfig,
    TFAutoModelForSequenceClassification,
    TFTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# region Helper functions


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


# endregion

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

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


# region Command-line arguments
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={
        "help":
        "The name of the task to train on: " + ", ".join(task_to_keys.keys())
    }, )
    predict_file: str = field(
        metadata={
            "help":
            "A file containing user-supplied examples to make predictions for"
        },
        default=None,
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached preprocessed datasets or not."
        })
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
            "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()
        if self.task_name not in task_to_keys.keys():
            raise ValueError("Unknown task, you should pick one in " +
                             ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
            "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help":
            "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
            "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    teacher_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    teacher_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    teacher_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    teacher_model_revision: str = field(
        default="main",
        metadata={
            "help":
            "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    teacher_use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
            "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to what type of optimization we are going to apply on the model.
    """

    autodistill: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply distillation."},
    )
    distillation_mode: Optional[str] = field(
        default="KnowledgeLoss",
        metadata={
            "help":
            "distillation criterion mode Supported approach is KnowledgeLoss."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature of distillation"},
    )
    loss_types: List[str] = field(
        default=None,
        metadata={"help": "Loss Type used for the distillation strategy."},
    )
    loss_weight_ratio: List[float] = field(
        default=None,
        metadata={
            "help": "Loss weight ratio used for the distillation strategy."
        },
    )
    layer_mappings: List[str] = field(
        default=None,
        metadata={"help": "Performance tolerance when optimizing the model."},
    )
    add_origin_loss: bool = field(
        default=False, metadata={"help": "Whether add the origin loss or not"})
    benchmark: bool = field(default=False, metadata={"help": "Run benchmark."})
    use_distillation_model: bool = field(
        default=False,
        metadata={"help":"Whether to use pretrained distillation model."})
    accuracy_only: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to only test accuracy for model tuned by Neural Compressor."
        })


@dataclass
class DistributedArguments:
    """
    Arguments pertaining to what type of distribute we are going to apply on the model.
    """
    multinode: bool = field(
        default=False,
        metadata={"help": "Whether to use the mutltinode mode."},
    )
    worker: str = field(
        default=None,
        metadata={"help": "List of node ip addressesg, using comma to split."},
    )
    task_index: int = field(
        default=0,
        metadata={
            "help":
            "Worker index, and worker with index 0 as the main worker for more works, such as saving model."
        },
    )


# endregion


def main():
    # region Argument parsing
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TFTrainingArguments,
         OptimizationArguments, DistributedArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, optim_args, distributed_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, optim_args, distributed_args = parser.parse_args_into_dataclasses(
        )

    if not (training_args.do_train or training_args.do_eval
            or training_args.do_predict):
        exit(
            "Must specify at least one of --do_train, --do_eval or --do_predict!"
        )
    # endregion

    # region Checkpoints
    checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(
                training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # endregion

    # region Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank
                                                    ) else logging.WARN)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    # endregion

    # region Dataset and labels
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub. In distributed training, the load_dataset function guarantee
    # that only one local process can concurrently download the dataset.
    datasets = load_dataset(
        "glue",
        data_args.task_name,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    if data_args.predict_file is not None:
        logger.info("Preparing user-supplied file for predictions...")

        data_files = {"data": data_args.predict_file}

        for key in data_files.keys():
            logger.info(f"Loading a local file for {key}: {data_files[key]}")

        if data_args.predict_file.endswith(".csv"):
            # Loading a dataset from local csv files
            user_dataset = load_dataset("csv",
                                        data_files=data_files,
                                        cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            user_dataset = load_dataset("json",
                                        data_files=data_files,
                                        cache_dir=model_args.cache_dir)
        needed_keys = task_to_keys[data_args.task_name]
        for key in needed_keys:
            assert key in user_dataset[
                "data"].features, f"Your supplied predict_file is missing the {key} key!"
        datasets["user_data"] = user_dataset["data"]
    # endregion

    # region Load model config and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # endregion

    # region Dataset preprocessing
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    non_label_column_names = [
        name for name in datasets["train"].column_names if name != "label"
    ]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if config.label2id != PretrainedConfig(
            num_labels=num_labels).label2id and not is_regression:
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]])
                for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
            label_to_id = {label: i for i, label in enumerate(label_list)}
    if label_to_id is not None:
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = ((examples[sentence1_key], ) if sentence2_key is None else
                (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*args,
                           padding=padding,
                           max_length=max_seq_length,
                           truncation=True)

        return result

    datasets = datasets.map(preprocess_function,
                            batched=True,
                            load_from_cache_file=not data_args.overwrite_cache)

    if data_args.pad_to_max_length:
        data_collator = DefaultDataCollator(return_tensors="tf")
    else:
        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")
    # endregion

    # region Metric function
    metric = load_metric("glue", data_args.task_name)

    def compute_metrics(preds, label_ids):
        preds = preds["logits"]
        preds = np.squeeze(preds) if is_regression else np.argmax(preds,
                                                                  axis=1)
        result = metric.compute(predictions=preds, references=label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # endregion
    if distributed_args.multinode:
        logger.info('*** using multinode mode... ***')
        from intel_extension_for_transformers.optimization.utils.utility_tf import distributed_init, get_filepath
        assert distributed_args.worker is not None, "worker address list should not be empty"
        distributed_args.worker = distributed_args.worker.strip().split(',')
        distributed_init(distributed_args.worker,
                        'worker',
                         distributed_args.task_index)

        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        training_args.output_dir = get_filepath(
            training_args.output_dir, strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id)
    else:
        strategy = training_args.strategy
    with strategy.scope():
        # region Load pretrained model
        if checkpoint is None:
            model_path = model_args.model_name_or_path
        else:
            model_path = checkpoint
        model = TFAutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # endregion

        # region Optimizer, loss and compilation
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=training_args.learning_rate,
            beta_1=training_args.adam_beta1,
            beta_2=training_args.adam_beta2,
            epsilon=training_args.adam_epsilon,
            clipnorm=training_args.max_grad_norm,
        )
        if is_regression:
            loss_fn = tf.keras.losses.MeanSquaredError()
            metrics = []
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
            metrics = ["accuracy"]
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        # endregion

        # region Convert data to a tf.data.Dataset
        tf_data = dict()
        max_samples = {
            "train": data_args.max_train_samples,
            "validation": data_args.max_eval_samples,
            "validation_matched": data_args.max_eval_samples,
            "validation_mismatched": data_args.max_eval_samples,
            "test": data_args.max_predict_samples,
            "test_matched": data_args.max_predict_samples,
            "test_mismatched": data_args.max_predict_samples,
            "user_data": None,
        }
        for key in datasets.keys():
            if key == "train" or key.startswith("validation"):
                assert "label" in datasets[
                    key].features, f"Missing labels from {key} data!"
            if key == "train":
                shuffle = True
                batch_size = training_args.per_device_train_batch_size * (
                    len(distributed_args.worker)
                    if distributed_args.worker is not None else 1)
                drop_remainder = True  # Saves us worrying about scaling gradients for the last batch
            else:
                shuffle = False
                batch_size = training_args.per_device_train_batch_size * (
                    len(distributed_args.worker)
                    if distributed_args.worker is not None else 1)
                drop_remainder = False
            samples_limit = max_samples[key]
            dataset = datasets[key]
            if samples_limit is not None:
                dataset = dataset.select(range(samples_limit))
            data = dataset.to_tf_dataset(
                columns=[
                    col for col in dataset.column_names
                    if col not in set(non_label_column_names + ["label"])
                ],
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=data_collator,
                drop_remainder=drop_remainder,
                # `label_cols` is needed for user-defined losses, such as in this example
                # datasets v2.3.x need "labels", not "label"
                label_cols=["labels"]
                if "label" in dataset.column_names else None,
            )
            tf_data[key] = data
        # endregion
    if optim_args.autodistill:
        logger.info('*** start distillation... ***')
        from intel_extension_for_transformers.optimization import metrics, TFOptimization, AutoDistillationConfig, TFDistillationConfig
        optimization = TFOptimization(
            model=model,
            args=training_args,
            train_dataset=tf_data["train"],
            eval_dataset=tf_data["validation"],
            compute_metrics=compute_metrics,
            task_type=strategy.cluster_resolver.task_type if distributed_args.multinode else None,
            task_id=strategy.cluster_resolver.task_id if distributed_args.multinode else None,
            strategy=strategy
            )
        
        # load teacher_model
        teacher_config = AutoConfig.from_pretrained(
            model_args.teacher_config_name
            if model_args.teacher_config_name else
            model_args.teacher_model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.teacher_cache_dir,
            revision=model_args.teacher_model_revision,
            use_auth_token=True
            if model_args.teacher_use_auth_token else None,
        )
        teacher_model = TFAutoModelForSequenceClassification.from_pretrained(
            model_args.teacher_model_name_or_path,
            config=teacher_config,
            cache_dir=model_args.teacher_cache_dir,
            revision=model_args.teacher_model_revision,
            use_auth_token=True
            if model_args.teacher_use_auth_token else None,
        )
        loss_type = [
            'CE', 'CE'
        ] if optim_args.loss_types is None else optim_args.loss_types
        loss_weight_ratio = [
            0.5, 0.5
        ] if optim_args.loss_weight_ratio is None else optim_args.loss_weight_ratio
        autodistillation_config = AutoDistillationConfig(
            search_space={
                'hidden_size': [120, 240],
                'intermediate_size': [256, 512]
            },
            search_algorithm='Grid',
            max_trials=3,
            metrics=[
                metrics.Metric(name="metric", greater_is_better=False)
            ],
            knowledge_transfer=TFDistillationConfig(
                train_steps=[3],
                loss_types=loss_type,
                loss_weights=loss_weight_ratio,
                temperature=optim_args.temperature
            ),
        )
        best_model_archs = optimization.autodistill(
            autodistillation_config,
            teacher_model,
            model_cls=TFAutoModelForSequenceClassification
        )

        return

    # region Training and validation
    if training_args.do_train:
        callbacks = [
            SavePretrainedCallback(output_dir=training_args.output_dir)
        ]
        if training_args.do_eval and not data_args.task_name == "mnli":
            # Do both evaluation and training in the Keras fit loop, unless the task is MNLI
            # because MNLI has two validation sets
            validation_data = tf_data["validation"]
        else:
            validation_data = None
        model.fit(
            tf_data["train"],
            validation_data=validation_data,
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

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        if data_args.task_name == "mnli":
            tasks = ["mnli", "mnli-mm"]
            tf_datasets = [
                tf_data["validation_matched"],
                tf_data["validation_mismatched"]
            ]
            raw_datasets = [
                datasets["validation_matched"],
                datasets["validation_mismatched"]
            ]
        else:
            tasks = [data_args.task_name]
            tf_datasets = [tf_data["validation"]]
            raw_datasets = [datasets["validation"]]

        total_time = 0
        num_examples = 0
        if optim_args.use_distillation_model:
            model = tf.saved_model.load(training_args.output_dir)
        for raw_dataset, tf_dataset, task in zip(raw_datasets, tf_datasets,
                                                    tasks):
            num_examples += sum(
                1 for _ in (tf_dataset.unbatch()
                            if hasattr(tf_dataset, "unbatch") else tf_dataset
                            )
            )

            if optim_args.use_distillation_model:
                preds: np.ndarray = None
                label_ids: np.ndarray = None
                infer = model.signatures[list(model.signatures.keys())[0]]
                for i, (inputs, labels) in enumerate(tf_dataset):
                    for name in inputs:
                        inputs[name] = tf.constant(inputs[name].numpy(), dtype=tf.int32)
                    start = time.time()
                    results = infer(**inputs)
                    total_time += time.time() - start
                    for val in results:
                        if preds is None:
                            preds = results[val].numpy()
                        else:
                            preds = np.append(preds, results[val].numpy(), axis=0)
                    if label_ids is None:
                        label_ids = labels.numpy()
                    else:
                        label_ids = np.append(label_ids, labels.numpy(), axis=0)
                eval_metrics = compute_metrics({"logits": preds}, label_ids)
            else:
                start = time.time()
                eval_predictions = model.predict(tf_dataset)
                total_time += time.time() - start
                eval_metrics = compute_metrics(eval_predictions, raw_dataset["label"])
                print(f"Evaluation metrics ({task}):")
                print(eval_metrics)
            logger.info("metric ({}) Accuracy: {}".format(task, eval_metrics["accuracy"]))
        logger.info(
           "Throughput: {} samples/sec".format(
                num_examples / total_time)
        )
    # endregion

    # region Prediction
    if training_args.do_predict or data_args.predict_file:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = []
        tf_datasets = []
        raw_datasets = []
        if training_args.do_predict:
            if data_args.task_name == "mnli":
                tasks.extend(["mnli", "mnli-mm"])
                tf_datasets.extend(
                    [tf_data["test_matched"], tf_data["test_mismatched"]])
                raw_datasets.extend([
                    datasets["test_matched"], datasets["test_mismatched"]
                ])
            else:
                tasks.append(data_args.task_name)
                tf_datasets.append(tf_data["test"])
                raw_datasets.append(datasets["test"])
        if data_args.predict_file:
            tasks.append("user_data")
            tf_datasets.append(tf_data["user_data"])
            raw_datasets.append(datasets["user_data"])

        if optim_args.use_distillation_model:
            model = tf.saved_model.load(training_args.output_dir)

        for raw_dataset, tf_dataset, task in zip(raw_datasets, tf_datasets,
                                                    tasks):
            if optim_args.use_distillation_model:
                preds: np.ndarray = None
                infer = model.signatures[list(model.signatures.keys())[0]]
                for i, (inputs, labels) in enumerate(tf_dataset):
                    for name in inputs:
                        inputs[name] = tf.constant(inputs[name].numpy(), dtype=tf.int32)
                    results = infer(**inputs)
                    for val in results:
                        if preds is None:
                            preds = results[val].numpy()
                        else:
                            preds = np.append(preds, results[val].numpy(), axis=0)
                test_predictions = {"logits": preds}
            else:
                test_predictions = model.predict(tf_dataset)
            if "label" in raw_dataset:
                test_metrics = compute_metrics(test_predictions,
                                                raw_dataset["label"])
                print(f"Test metrics ({task}):")
                print(test_metrics)

            if is_regression:
                predictions_to_write = np.squeeze(
                    test_predictions["logits"])
            else:
                predictions_to_write = np.argmax(
                    test_predictions["logits"], axis=1)

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt")
            with open(output_predict_file, "w") as writer:
                logger.info(
                    f"***** Writing prediction results for {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions_to_write):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = config.id2label[item]
                        writer.write(f"{index}\t{item}\n")
    # endregion


if __name__ == "__main__":
    main()
