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

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Optional
from scipy.special import softmax
import yaml
import numpy as np

SEC_TO_NS_SCALE = 1000000000


@dataclass
class Benchmark:
    summary_msg: str = field(default_factory=str)

    @property
    def num_runs(self) -> int:
        return len(self.latencies)

    @contextmanager
    def track(self, step):
        start = perf_counter_ns()
        yield
        ns = perf_counter_ns() - start
        msg = f"\n{'*' * 70}\n'{step}' took {ns / SEC_TO_NS_SCALE:.3f}s ({ns:,}ns)\n{'*' * 70}\n"
        # print(msg)
        self.summary_msg += msg + '\n'

    def summary(self):
        print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")


@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    smoke_test: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to execute in sanity check mode."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker testing, truncate the number of testing examples to this "
                    "value if set."
        },
    )
    instance_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "for multi-instance inference, to indicate which instance this is."
        },
    )
    dataset: Optional[str] = field(
        default='imdb',
        metadata={
            "help": "Select dataset ('imdb' / 'sst2'). Default is 'imdb'"
        },
    )
    max_seq_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    profiler: int = field(
        default=0,
        metadata={
            "help": "whether using pytorch profiler"
        },
    )
    profiler_name: str = field(
        default="test",
        metadata={
            "help": "log name for pytorch profiler"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    use_tpp: bool = field(
        default=False,
        metadata={
            "help": "Use TPP Extension for PyTorch for fine-Tuning."
        },
    )
    unpad: bool = field(
        default=False,
        metadata={
            "help": "Use TPP Extension for PyTorch for fine-Tuning."
        },
    )
    """
    Arguments for test scenarios
    """
    infer_impl: Optional[str] = field(
        default="trainer", metadata={
            "help": "The implementation of inference pipeline. Now we support trainer and ipex implementation."
        }
    )

    finetune_impl: Optional[str] = field(
        default="trainer", metadata={
            "help": "The implementation of fine-tuning pipeline. Now we support trainer and ipex implementation."
        }
    )

    dtype_inf: Optional[str] = field(
        default="fp32", metadata={
            "help": "Data type for inference pipeline."
                    "Support fp32, bf16, and int8 for CPU. Support fp32 and fp16 for GPU "
        }
    )
    dtype_ft: Optional[str] = field(
        default="fp32", metadata={
            "help": "Data type for finetune pipeline."
                    "Support fp32 and bf16 for CPU. Support fp32, tf32, and fp16 for GPU "
        }
    )
    multi_instance: bool = field(
        default=False,
        metadata={
            "help": "Whether to use multi-instance mode"
        },
    )
    dist_backend: Optional[str] = field(
        default="ccl", metadata={"help": "Distributed backend to use for fine-tuning"}
    )
    save_detailed_performance_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to save performance metrics or not"}
    )


class PredsLabels:
    def __init__(self, preds, labels):
        self.predictions = preds
        self.label_ids = labels


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def save_train_metrics(train_result, trainer, max_train):
    # pytorch only
    if train_result:
        metrics = train_result.metrics
        metrics["train_samples"] = max_train
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def save_test_metrics(metrics, max_test, output_dir):
    metrics['test_samples'] = max_test
    with open(Path(output_dir) / 'test_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return "\n\n******** TEST METRICS ********\n" + '\n'.join(f'{k}: {v}' for k, v in metrics.items())


def save_performance_metrics(trainer, data, output_file):
    label_map = {i:v for i,v in enumerate(data.features['label'].names)}
    predictions = trainer.predict(data)

    predictions_report = {}
    predictions_report["label_id"] = [label_map[i] for i in predictions.label_ids.tolist()]
    predictions_report["predictions_label"] = [label_map[i] for i in np.argmax(predictions.predictions, axis=1).tolist() ] 
    predictions_report["predictions_probabilities"] = softmax(predictions.predictions, axis=1).tolist() 
    predictions_report["metrics"] = predictions.metrics
    
    with open(output_file, 'w') as file:
        _ = yaml.dump(predictions_report, file) 


