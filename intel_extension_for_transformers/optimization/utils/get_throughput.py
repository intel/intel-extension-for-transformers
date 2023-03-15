#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

"""Special file for triggering multiple instances for benchmarking."""
import argparse
import time
import torch
from collections import UserDict
from neural_compressor.utils import logger
from intel_extension_for_transformers.optimization import OptimizedModel


def torch_inference(model, data):
    start_time = time.time()
    with torch.no_grad():
        if isinstance(data, dict) or isinstance(data, UserDict):
            model(**data)
        elif isinstance(data, list) or isinstance(data, tuple):
            model(*data)
        else:
            model(data)
    return time.time() - start_time


def get_latency_per_iter(model, data, warmup, iters):
    time_usage = 0
    for i in range(iters):
        if i < warmup:
            torch_inference(model, data)
        else:
            time_usage += torch_inference(model, data)
    return time_usage / (iters - warmup)


if __name__ == "__main__":
    logger.info("Evaluating performance:")
    parser = argparse.ArgumentParser(description="Evaluate performance")
    parser.add_argument("--model", type=str, help="input model")
    parser.add_argument("--data", type=str, help="example inputs")
    parser.add_argument("--batch_size", default=1, type=int, help="the batch size of input data")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="total iterations")
    parser.add_argument("--torchscript", action='store_true', help="the model is torchscript")
    parser.add_argument("--generate", action='store_true', help="evaluate model.generate performance")
    parser.add_argument("--from_pretrain", action='store_true', help="load model ")
    parser.add_argument("--enable_ipex", action='store_true', help="load model ")
    args = parser.parse_args()
    if args.enable_ipex:
        import intel_extension_for_pytorch as ipex
    if args.torchscript:
        model = torch.jit.load(args.model)
    else:
        if args.from_pretrain:
            model = OptimizedModel.from_pretrained(args.model)
        else:
            model = torch.load(args.model)
    model.eval()
    if args.generate:
        model = model.generate
    data = torch.load(args.data)
    time_usage = get_latency_per_iter(model, data, args.warmup, args.iters)
    latency = time_usage / args.batch_size
    throughput = args.batch_size / time_usage
    logger.info("Batch size: {}".format(args.batch_size))
    logger.info("Latency: {:.4f} second/sample".format(latency))
    logger.info("Throughput: {:.4f} samples/second".format(throughput))
