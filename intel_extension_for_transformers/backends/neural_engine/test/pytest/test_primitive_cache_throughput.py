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

import time
import unittest
import numpy as np
import os
import shutil
import subprocess
import platform
import torch
from transformers import BertForSequenceClassification
from intel_extension_for_transformers.backends.neural_engine.compile import compile


class TestPrimitiveCacheThroughput(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.skipTest(self, "currently not support Unit Test for dispatcher, but this function is supported. Will improve Unit Test very soon.")
        if platform.system().lower() == "windows":
            self.skipTest(self, "not support on windows yet")
        code = """
import time
import math
import os
import sys
import random
import numpy as np
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph


class MRPCDataSet():
    def __init__(self, batch_size, data_dir, tokenizer_dir):
        self.batch_size = batch_size
        self.dataset = []
        for j in range(3000):
            seq_len = random.randint(1, 128)
            input_ids = np.random.randint(low = 1, high = 10000,
                                        size = (batch_size, seq_len), dtype = np.int32)
            segment_ids = np.random.randint(low = 0, high = 2,
                                        size = (batch_size, seq_len), dtype = np.int32)
            input_mask = np.random.randint(low = 1, high = 2,
                                        size = (batch_size, seq_len), dtype = np.int32)
            label = np.ones((batch_size, 1))
            self.dataset.append([input_ids, segment_ids, input_mask, label])

    def __getitem__(self, idx):
        input_ids_data = self.dataset[idx][0]
        segment_ids_data = self.dataset[idx][1]
        input_mask_data = self.dataset[idx][2]
        label_data = self.dataset[idx][3]

        return [input_ids_data,
                segment_ids_data,
                input_mask_data], label_data

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

def load_model(engine_model_path):
    model = Graph()
    model.graph_init(os.path.join(engine_model_path, "conf.yaml"),
                    os.path.join(engine_model_path, "model.bin"), load_weight=True)
    return model

def run():
    os.environ['GLOG_minloglevel']='2'
    # cycle buffer
    if os.environ.get('DIRECT_BUFFER'):
        del os.environ['DIRECT_BUFFER']
    if os.environ.get('UNIFIED_BUFFER'):
        del os.environ['UNIFIED_BUFFER']
    data_path = "/home/tensorflow/.cache/nlp_toolkit/bert_mini_mrpc"
    model_path = "/tf_dataset2/models/nlp_toolkit/bert_mini_mrpc"
    dataset = MRPCDataSet(1, data_path, model_path)
    model = load_model("ir")
    time_list = []
    log_path = sys.argv[1]
    for idx in range(len(dataset)):
        inputs = dataset[idx][0]
        t1 = time.time()
        out = model.inference(inputs)
        time_list.append(time.time() - t1)

    latency = np.array(time_list[300:]).mean()
    throughput = int(1 * 1 / latency)
    with open(log_path, 'w') as f:
        f.write(str(throughput))

if __name__ == "__main__":
    run()

"""
        with open('run.py', 'w', encoding='utf-8') as f:
            f.write(code)
        # export onnx model
        model_path = "/tf_dataset2/models/nlp_toolkit/bert_mini_mrpc"
        torch_model = BertForSequenceClassification.from_pretrained(model_path)
        with torch.no_grad():
            inputs = {
                'input_ids': torch.ones(1, 128, dtype=torch.int32),
                'attention_mask': torch.ones(1, 128, dtype=torch.int32),
                'token_type_ids': torch.ones(1, 128, dtype=torch.int32)
            }
            outputs = torch_model(**inputs)

            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(
                torch_model,
                (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']),
                "onnx_fp32.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'input_mask', 'segment_ids'],
                output_names=['output'],
                dynamic_axes={
                    'input_ids': symbolic_names,
                    'input_mask': symbolic_names,
                    'segment_ids': symbolic_names
                })
        graph = compile("onnx_fp32.onnx")
        graph.save()

    @classmethod
    def tearDownClass(self):
        os.remove("run.py")
        os.remove("onnx_fp32.onnx")
        shutil.rmtree("./ir", ignore_errors=True)
        for i in range(4):
            try:
                os.remove("log" + str(i) + "_pc0.txt")
                os.remove("log" + str(i) + "_pc1.txt")
            except:
                continue

    def test_primitive_cache_throughput(self):
        cmd = "numactl -l -C 0-6 python run.py  log0_pc0.txt & " \
              "numactl -l -C 7-13 python run.py log1_pc0.txt & " \
              "numactl -l -C 14-20 python run.py log2_pc0.txt & " \
              "numactl -l -C 21-27 python run.py log3_pc0.txt"
        # close engine primitive_cache
        os.environ['ENGINE_PRIMITIVE_CACHE_OFF'] = '1'
        os.environ['INST_NUM'] = '4'
        os.environ['GLOG_minloglevel'] = '2'
        # cycle buffer
        if os.environ.get('DIRECT_BUFFER'):
            del os.environ['DIRECT_BUFFER']
        if os.environ.get('UNIFIED_BUFFER'):
            del os.environ['UNIFIED_BUFFER']

        process = subprocess.Popen(cmd, shell=True)  # nosec
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

        # wait all threads end
        for i in range(4):
            log_exist = os.path.exists("log" + str(i) + "_pc0.txt")
            time_exit = 0
            while not log_exist:
                time.sleep(1)
                time_exit += 1
                log_exist = os.path.exists("log" + str(i) + "_pc0.txt")
                if time_exit >= 600:
                    break

        throughput_off = 0
        for i in range(4):
            with open("log" + str(i) + "_pc0.txt", 'r') as f:
                throughput_off += int(f.readline().strip())

        # open engine primitive_cache
        del os.environ['ENGINE_PRIMITIVE_CACHE_OFF']
        cmd = "numactl -l -C 0-6 python run.py  log0_pc1.txt & " \
              "numactl -l -C 7-13 python run.py log1_pc1.txt & " \
              "numactl -l -C 14-20 python run.py log2_pc1.txt & " \
              "numactl -l -C 21-27 python run.py log3_pc1.txt"
        process = subprocess.Popen(cmd, shell=True)  # nosec
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)

        # wait all threads end
        for i in range(4):
            log_exist = os.path.exists("log" + str(i) + "_pc1.txt")
            time_exit = 0
            while not log_exist:
                time.sleep(1)
                time_exit += 1
                log_exist = os.path.exists("log" + str(i) + "_pc1.txt")
                if time_exit >= 600:
                    break

        throughput_on = 0
        for i in range(4):
            with open("log" + str(i) + "_pc1.txt", 'r') as f:
                throughput_on += int(f.readline().strip())

        self.assertGreater(throughput_on, throughput_off)


if __name__ == "__main__":
    unittest.main()
