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
import numpy as np
from nlp_toolkit.backends.neural_engine.compile import compile
from nlp_toolkit.backends.neural_engine.compile.graph import Graph

# set log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("NEURAL-ENGINE--EXAMPLES")

# set log file 
def set_log_file(log, log_file):
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        "%Y-%m-%d %H:%M:%S")
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
            front, ext= os.path.splitext(file_name)
            if ext == ".yaml":
                yaml_path = os.path.join(model_path, file_name)
            elif ext == ".bin":
                bin_path = os.path.join(model_path, file_name)
            else:
                log.error(
                    "IR directory should only has yaml and bin."
                )
                raise ValueError()
        graph = Graph()
        graph.graph_init(yaml_path, bin_path)
    else:
        log.error(
            "IR directory should only has 2 files."
        )
        raise ValueError()

    return graph

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
