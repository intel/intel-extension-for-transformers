#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from executor_utils import log, Neural_Engine


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        default="./model_and_tokenizer/int8-model.onnx",
                        type=str,
                        help="Input model path.")
    parser.add_argument("--mode",
                        default="accuracy",
                        type=str,
                        help="Benchmark mode of performance or accuracy.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--seq_len", default=128, type=int, help="Sequence length.")
    parser.add_argument("--warm_up",
                        default=5,
                        type=int,
                        help="Warm up iteration in performance mode.")
    parser.add_argument("--iteration", default=10, type=int, help="Iteration in performance mode.")
    parser.add_argument("--tokenizer_dir",
                        default="gchhablani/bert-base-cased-finetuned-mrpc",
                        type=str,
                        help="Pre-trained model tokenizer name or path")
    parser.add_argument("--data_dir", default="./data", type=str, help="Data cache directory.")
    parser.add_argument("--dataset_name", default="glue", type=str, help="Name of dataset.")
    parser.add_argument("--task_name", default="mrpc", type=str, help="Task name of dataset.")
    parser.add_argument("--log_file",
                        default="executor.log",
                        type=str,
                        help="File path to log information.")
    parser.add_argument("--dynamic_quantize",
                        default=False,
                        type=bool,
                        help="dynamic quantize for fp32 model.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.dynamic_quantize:
        executor = Neural_Engine(args.input_model, args.log_file, "dynamic_int8")
    else:
        executor = Neural_Engine(args.input_model, args.log_file, "native")
    if args.mode == "accuracy":
        executor.accuracy(args.batch_size, args.seq_len, args.dataset_name, args.task_name,
                          args.data_dir, args.tokenizer_dir)
    elif args.mode == "performance":
        executor.performance(args.batch_size, args.seq_len, args.iteration, args.warm_up)
    else:
        log.error("Benchmark only has performance or accuracy mode")
