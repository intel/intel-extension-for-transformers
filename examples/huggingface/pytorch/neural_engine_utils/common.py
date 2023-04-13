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

import os
import logging
import sys
import numpy as np
from intel_extension_for_transformers.backends.neural_engine.compile import compile, autocast
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from tqdm import tqdm
import time

# set log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("NEURAL-ENGINE--EXAMPLES")


# set log file
def set_log_file(log, log_file):
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)


#load graph
def load_graph(model_path):
    if os.path.exists(model_path):
        if os.path.isdir(model_path):
            graph = load_graph_from_ir(model_path)
        else:
            graph = compile(model_path)
    else:
        log.error("Model path doesn't exist.")
        raise ValueError()

    return graph


def load_graph_from_ir(model_path):
    file_list = os.listdir(model_path)
    if len(file_list) == 2:
        for file_name in file_list:
            front, ext = os.path.splitext(file_name)
            if ext == ".yaml":
                yaml_path = os.path.join(model_path, file_name)
            elif ext == ".bin":
                bin_path = os.path.join(model_path, file_name)
            else:
                log.error("IR directory should only has yaml and bin.")
                raise ValueError()
        graph = Graph()
        graph.graph_init(yaml_path, bin_path)
    else:
        log.error("IR directory should only has 2 files.")
        raise ValueError()

    return graph


def compute_performance(dataset, graph, log, log_file, warm_up, batch_size, seq_len):
    log.info("Start executor ......")
    duration = []
    for idx in tqdm(range(len(dataset))):
        start_time = time.time()
        predictions = graph.inference(dataset[idx])
        end_time = time.time()
        duration.append(end_time - start_time)
    log.info("End executor ......")
    duration_w = duration[warm_up:]
    all_latency = log_file.replace('.log', '.npy')
    _, file_name = os.path.split(all_latency)
    _ = os.getcwd() + '/all_latency'
    try:
        if os.path.exists(_) == False:
            os.mkdir(_)
    except:
        pass
    all_latency = os.path.join(_, file_name)
    All_latency = np.array(duration_w)
    np.save(all_latency, All_latency, allow_pickle=True, fix_imports=True)
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


# dummy dataloader
class DummyDataLoader(object):

    def __init__(self, shapes, lows, highs, dtypes, iteration):
        self.iteration = iteration
        self.dataset = []
        for _ in range(iteration):
            datas = []
            for i in range(len(shapes)):
                shape = shapes[i]
                low = lows[i]
                high = highs[i]
                dtype = dtypes[i]
                data = np.random.uniform(low=low, high=high, size=shape).astype(dtype)
                datas.append(data)
            self.dataset.append(datas)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.iteration


class Neural_Engine_base():

    def __init__(self, model_path, log_file, cast_type="native"):
        set_log_file(log, log_file)
        with autocast(cast_type):
            self.graph = compile(model_path)
        self.log_file = log_file

    def accuracy(self, batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir):
        pass

    def performance(self, batch_size, seq_len, iteration, warm_up):
        pass
