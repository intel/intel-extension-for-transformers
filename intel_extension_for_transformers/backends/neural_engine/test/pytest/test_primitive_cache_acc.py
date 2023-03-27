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

import unittest
import numpy as np
import os
import math
import sys
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import BertForSequenceClassification
from intel_extension_for_transformers.backends.neural_engine.compile import compile


class MRPCDataSet():

    def __init__(self, batch_size, data_dir, tokenizer_dir):
        self.batch_size = batch_size
        dataset = load_dataset('glue', 'mrpc', cache_dir=data_dir, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.dataset = dataset.map(lambda e: tokenizer(
            e['sentence1'], e['sentence2'], truncation=False, padding='do_not_pad'),
                                   batched=True)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        if end > len(self.dataset):
            input_ids_data = self.dataset[start:]['input_ids']
            segment_ids_data = self.dataset[start:]['token_type_ids']
            input_mask_data = self.dataset[start:]['attention_mask']
            label_data = self.dataset[start:]['label']
        else:
            input_ids_data = self.dataset[start:end]['input_ids']
            segment_ids_data = self.dataset[start:end]['token_type_ids']
            input_mask_data = self.dataset[start:end]['attention_mask']
            label_data = self.dataset[start:end]['label']

        sample_size = len(input_ids_data) if isinstance(input_ids_data, list) else 1

        return [np.array(input_ids_data).reshape(sample_size, -1).astype('int32'),
                np.array(segment_ids_data).reshape(sample_size, -1).astype('int32'),
                np.array(input_mask_data).reshape(sample_size, -1).astype('int32')], \
                np.array(label_data).reshape(sample_size, -1).astype('int32')

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def load_model(model_name, onnx_model_path):
    torch_model = BertForSequenceClassification.from_pretrained(model_name)
    with torch.no_grad():
        inputs = {
            'input_ids': torch.ones(1, 128, dtype=torch.int32),
            'attention_mask': torch.ones(1, 128, dtype=torch.int32),
            'token_type_ids': torch.ones(1, 128, dtype=torch.int32)
        }
        outputs = torch_model(**inputs)

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            torch_model, (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']),
            onnx_model_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'input_mask', 'segment_ids'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': symbolic_names,
                'input_mask': symbolic_names,
                'segment_ids': symbolic_names
            })
    graph = compile(onnx_model_path)
    return graph


class TestPrimitiveCacheAcc(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.skipTest(self, "currently not support Unit Test for dispatcher, but this function is supported. Will improve Unit Test very soon.")
        is_win = sys.platform.startswith('win')
        self.model_name = '/tf_dataset2/models/nlp_toolkit/bert_mini_mrpc'
        self.data_dir = '/home/tensorflow/.cache/nlp_toolkit/bert_mini_mrpc'
        if is_win:
            self.model_name = "D:\\dataset\\models\\pytorch\\bert_mini_mrpc"
            self.data_dir = "D:\\dataset\\glue_data\\bert_mini_mrpc_data"
        self.onnx_model_path = 'onnx_fp32.onnx'

    @classmethod
    def tearDownClass(self):
        os.remove(self.onnx_model_path)

    def test_primitive_cache_acc(self):
        os.environ['GLOG_minloglevel'] = '2'
        metric = load_metric('glue', 'mrpc')
        batch_size = 1
        dataset = MRPCDataSet(batch_size, self.data_dir, self.model_name)
        model = load_model(self.model_name, self.onnx_model_path)
        # close engine primitive_cache
        os.environ['ENGINE_PRIMITIVE_CACHE_OFF'] = '1'
        # cycle buffer
        if os.environ.get('DIRECT_BUFFER'):
            del os.environ['DIRECT_BUFFER']
        if os.environ.get('UNIFIED_BUFFER'):
            del os.environ['UNIFIED_BUFFER']
        for idx in range(len(dataset)):
            inputs = dataset[idx][0]
            labels = dataset[idx][1]
            predictions = model.inference(inputs)
            predictions = list(predictions.values())[0]
            predictions = np.argmax(predictions, axis=1)
            metric.add_batch(
                predictions=predictions,
                references=labels,
            )
        # compute metrics
        eval_metric = metric.compute()
        acc_off = eval_metric.get("accuracy")

        # open engine primitive_cache
        del os.environ['ENGINE_PRIMITIVE_CACHE_OFF']
        for idx in range(len(dataset)):
            inputs = dataset[idx][0]
            labels = dataset[idx][1]
            predictions = model.inference(inputs)
            predictions = list(predictions.values())[0]
            predictions = np.argmax(predictions, axis=1)
            metric.add_batch(
                predictions=predictions,
                references=labels,
            )
        # compute metrics
        eval_metric = metric.compute()
        acc_on = eval_metric.get("accuracy")
        # acc_on should equal to acc_off
        self.assertEqual(acc_off, acc_on)


if __name__ == "__main__":
    unittest.main()
