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

import math
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

class DataLoader(object):
    def __init__(self, batch_size, seq_len, dataset_name, task_name, data_dir, tokenizer_dir):
        self.batch_size = batch_size
        dataset = load_dataset(dataset_name, task_name, cache_dir=data_dir, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.dataset = dataset.map(lambda e: tokenizer(e['sentence'],
                    truncation=True, padding='max_length', max_length=seq_len), batched=True)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        if end > len(self.dataset):
            input_ids_data = self.dataset[start:]['input_ids']
            input_mask_data = self.dataset[start:]['attention_mask']
            label_data = self.dataset[start:]['label']
        else:
            input_ids_data = self.dataset[start:end]['input_ids']
            input_mask_data = self.dataset[start:end]['attention_mask']
            label_data = self.dataset[start:end]['label']

        sample_size = len(input_ids_data) if isinstance(input_ids_data, list) else 1
        
        return [np.array(input_ids_data).reshape(sample_size, -1).astype('int32'),
                np.array(input_mask_data).reshape(sample_size, -1).astype('int32')], \
                np.array(label_data).reshape(sample_size, -1).astype('int32')

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)
