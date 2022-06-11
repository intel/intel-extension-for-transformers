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

import time
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from executor_dataloader import DataLoader
from examples.deployment.neural_engine.common import (
    log, 
    set_log_file,
    load_graph, 
    DummyDataLoader
)

class Neural_Engine(object):
    def __init__(self, model_path, log_file):
        set_log_file(log, log_file)
        self.graph = load_graph(model_path)

    def accuracy(self, batch_size, seq_len, 
                 dataset_name, task_name, data_dir, tokenizer_dir):
        # load dataset
        log.info("Load dataset ......")
        dataset = DataLoader(batch_size, seq_len, dataset_name,
            task_name, data_dir, tokenizer_dir)
        # load metric
        log.info("Load metric ......")
        if dataset_name and task_name is not None:
            metric = load_metric(dataset_name, task_name)
        else:
            metric = load_metric("accuracy")
        # execute
        log.info("Start engine ......")
        for idx in tqdm(range(len(dataset))):
            inputs = dataset[idx][0]
            labels = dataset[idx][1]
            predictions = self.graph.inference(inputs)
            predictions = list(predictions.values())[0]
            predictions = np.argmax(predictions, axis=1)
            metric.add_batch(
                    predictions=predictions,
                    references=labels,
                )
        # compute metrics
        log.info("Compute metrics ......")
        eval_metric = metric.compute()
        accuracy_value = eval_metric.get("accuracy")
        f1_value = eval_metric.get("f1")
        log.info(f"Accuracy: {accuracy_value}")
        log.info(f"F1: {f1_value}")

    def performance(self, batch_size, seq_len, iteration, warm_up):
        if warm_up >= iteration:
            log.error("Warm up should less than iteration.")
            raise ValueError()
        # generate dummy dataset
        log.info("Generate dummy dataset ......")
        shape = [batch_size, seq_len]
        dataset = DummyDataLoader(shapes=[shape,  shape],
                                 lows=[0,  0],
                                 highs=[128,  1],
                                 dtypes=['int32', 'int32'],
                                 iteration=iteration)
        # execute
        log.info("Start engine ......")
        duration = []
        for idx in tqdm(range(len(dataset))):
            start_time = time.time()
            predictions = self.graph.inference(dataset[idx])
            end_time = time.time()
            duration.append(end_time - start_time)
        log.info("End engine ......")
        duration_w = duration[warm_up:]
        ave_latency = np.array(duration_w).mean() / batch_size
        p50_latency = np.percentile(duration_w, 50) / batch_size
        p90_latency = np.percentile(duration_w, 90) / batch_size
        p99_latency = np.percentile(duration_w, 99) / batch_size
        log.info("Batch size = {}".format(batch_size))
        log.info("Sequence length = {}".format(seq_len))
        log.info("P50 Latency: {:.3f} ms".format(p50_latency * 1000))
        log.info("P90 Latency: {:.3f} ms".format(p90_latency * 1000))
        log.info("P99 Latency: {:.3f} ms".format(p99_latency * 1000))
        log.info("Average Latency: {:.3f} ms".format(ave_latency * 1000))
        log.info("Throughput: {:.3f} samples/sec".format(1. / ave_latency))
