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
from intel_extension_for_transformers.transformers.runtime.compile import compile, autocast
from intel_extension_for_transformers.transformers.runtime.compile.graph import Graph
import sys
import os
import logging

common_dir = os.path.join(sys.path[0], "../../../../neural_engine_utils/")
sys.path.append(common_dir)
from common import (log, DummyDataLoader, compute_performance)


# set log file
def set_log_file(log, log_file):
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

class Neural_Engine_bge():

    def __init__(self, model_path, log_file, cast_type="native"):
        set_log_file(log, log_file)
        with autocast(cast_type):
            self.graph = compile(model_path)
        self.log_file = log_file

    def accuracy(self, batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir):
        pass

    def performance(self, batch_size, seq_len, iteration, warm_up):
        pass


class Neural_Engine(Neural_Engine_bge):

    def accuracy(self, batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir):
        # load dataset
        log.info("Load dataset ......")
        dataset = DataLoader(batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir)
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
        dataset = DummyDataLoader(shapes=[shape, shape, shape],
                                  lows=[1, 1, 1],
                                  highs=[128, 1, 1],
                                  dtypes=['int32', 'int32', 'int32'],
                                  iteration=iteration)
        compute_performance(dataset, self.graph, log, self.log_file, warm_up, batch_size, seq_len)
