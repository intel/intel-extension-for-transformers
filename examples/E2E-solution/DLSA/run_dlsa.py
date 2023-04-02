# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

"""E2E DLSA fine-tuning and inference pipeline with ITREX"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset

from neural_compressor.benchmark import fit
from neural_compressor.config import BenchmarkConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging as hf_logging
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
)
from intel_extension_for_transformers.optimization import (
    OptimizedModel,
    QuantizationConfig,
    metrics,
    objectives,
)
from intel_extension_for_transformers.optimization.trainer import NLPTrainer

hf_logging.set_verbosity_info()


@dataclass
class PredsLabels:
    """Class for the labels of the predictions"""

    def __init__(self, preds, labels):
        self.predictions = preds
        self.label_ids = labels


@dataclass
class DlsaPipeline:
    """Class for the E2E DlsaPipeline"""

    summary_msg: str = field(default_factory=str)
    sec_to_ns_scale: int = 1000000000

    @contextmanager
    def track(self, step):
        """Function tracking the elapsed time for each phase in the Benchmark"""
        start = perf_counter_ns()
        yield
        ns = perf_counter_ns() - start  # pylint: disable=C0103
        msg = f"\n{'*' * 70}\n'{step}' took {ns / self.sec_to_ns_scale:.3f}s ({ns:,}ns)\n{'*' * 70}\n"
        # print(msg)
        self.summary_msg += msg + "\n"

    def summary(self):
        """Function printing the Benchmark Summary"""
        print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="distilbert-base-uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name: Optional[str] = field(
        default="distilbert-base-uncased",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, "
            "truncate the number of training examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, "
            "truncate the number of testing examples to this value if set."
        },
    )
    dataset: Optional[str] = field(
        default="sst2",
        metadata={"help": "Select dataset ('imdb' / 'sst2'). Default is 'sst2'"},
    )
    max_seq_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_quantize: bool = field(
        default=False,
        metadata={"help": "Whether or not to apply quantization."},
    )
    do_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not to conduct inference benchmark."},
    )
    dtype_inf: Optional[str] = field(
        default="fp32",
        metadata={
            "help": "Data type for inference pipeline. Support fp32 and int8 now"
        },
    )
    num_of_instance: int = field(
        default=2,
        metadata={
            "help": "The instance number for benchmark. By default 4 cores per instance."
        },
    )


def compute_metrics(p):  # pylint: disable=C0103
    """Function calculating the total inference accuracy"""

    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def save_train_metrics(train_result, trainer, max_train):
    """Function saving the fine-tuning results"""

    # pytorch only
    if train_result:
        train_metrics = train_result.metrics
        train_metrics["train_samples"] = max_train
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()


def predict(model, trainer):
    """Prediction/evaluation loop"""

    batch_size = trainer.args.per_device_eval_batch_size
    all_outputs, all_labels = [], []

    def prediction_step(batch, labels):
        all_labels.extend(labels)
        inputs = batch
        output = model(**inputs)
        all_outputs.append(output["logits"])

    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            DataLoader(
                trainer.eval_dataset,
                batch_size=batch_size,
                collate_fn=DataCollatorWithPadding(trainer.tokenizer),
            )
        ):
            prediction_step(batch=batch, labels=batch.pop("labels"))

        acc = compute_metrics(
            PredsLabels(preds=np.concatenate(all_outputs), labels=all_labels)
        )
    return acc["acc"]


def main():
    """Function running the E2E DLSA pipeline"""
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # args = HfArgumentParser(Arguments).parse_args_into_dataclasses()
    # training_args = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    output_dir = Path(training_args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dlsaPipeline = DlsaPipeline()
    track = dlsaPipeline.track

    # pylint: disable=E1101
    max_train, max_test = args.max_train_samples, args.max_test_samples

    ################################# Load Data #################################

    with track("Load Data"):
        data = load_dataset(args.dataset)
        train_all = data["train"]
        test_split = "validation" if args.dataset == "sst2" else "test"
        len_train = len(train_all)
        train_data = (
            train_all.select(range(len_train - max_train, len_train))
            if max_train
            else train_all
        )
        test_data = (
            data[test_split].select(range(max_test)) if max_test else data[test_split]
        )
        text_column = [
            c for c in test_data.column_names if not isinstance(test_data[c][0], int)
        ][0]

    ################################# Pre-process #################################

    with track("Pre-process"):
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        )

        max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)

        def preprocess(examples):
            return tokenizer(
                examples[text_column],
                padding="max_length",
                truncation=True,
                max_length=max_seq_len,
            )

        kwargs = {
            "function": preprocess,
            "batched": True,
            "remove_columns": [text_column]
            + (["idx"] if args.dataset == "sst2" else []),
        }

        train_data = train_data.map(**kwargs)
        test_data = test_data.map(**kwargs)

    ################################# Load Model #################################

    with track("Load Model"):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path  # pylint: disable=E1101
        )

        trainer = NLPTrainer(
            model=model,  # the instantiated HF model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_data,  # training dataset
            eval_dataset=test_data,
            compute_metrics=compute_metrics,  # evaluation metrics
            tokenizer=tokenizer,
        )

        eval_dataloader = trainer.get_eval_dataloader(
            # using part of the test dataset for evaluation
            eval_dataset=(test_data.select(range(872))).remove_columns(
                column_names="label"
            )
        )

    ################################ Fine-Tune #################################

    if training_args.do_train:
        with track("Fine-Tune"):
            train_result = trainer.train()
            trainer.save_model()
            save_train_metrics(train_result, trainer, len(train_data))

    ################################ Quantize #################################

    if args.do_quantize:
        with track("Quantize"):
            metric = metrics.Metric(name="eval_acc", is_relative=True, criterion=0.01)
            q_config = QuantizationConfig(
                framework="pytorch_ipex",
                approach="PostTrainingStatic",
                max_trials=200,  # set the Max tune times
                metrics=[metric],
                objectives=[objectives.performance],
            )

            def eval_func(model):
                return predict(model, trainer)

            model = trainer.quantize(
                quant_config=q_config,
                calib_dataloader=eval_dataloader,
                eval_func=eval_func,
            )

    ############################## Inference #################################

    if training_args.do_predict:
        with track("Inference with Default FP32 Model"):
            inf_metrics = predict(trainer.model, trainer)
            print(f"\n*********** TEST_METRICS ***********\nAccuracy: {inf_metrics}\n")

        with track("Inference with ITREX Quantized INT8 Model"):
            inf_metrics = predict(model, trainer)
            print(f"\n*********** TEST_METRICS ***********\nAccuracy: {inf_metrics}\n")

    dlsaPipeline.summary()

    ############################## Benchmark #################################
    if args.do_benchmark:
        if args.dtype_inf == "int8":
            # Load the model obtained after Intel Neural Compressor (INC) quantization
            model = OptimizedModel.from_pretrained(args.model_name_or_path)
            trainer.model = model

        conf = BenchmarkConfig(
            warmup=10,
            iteration=100,
            cores_per_instance=4,
            num_of_instance=args.num_of_instance,
            backend="ipex",
        )
        fit(model=trainer.model, config=conf, b_dataloader=eval_dataloader)


if __name__ == "__main__":
    main()
