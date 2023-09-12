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

import argparse
import numpy as np
from intel_extension_for_transformers.neural_engine_py import Model

parser = argparse.ArgumentParser(description='Deep engine Model Executor')
parser.add_argument('--weight',  default='', type=str, help='weight of the model')
parser.add_argument('--config',  default='', type=str, help='config of the model')
args = parser.parse_args()

input_0 = np.random.randint(0,384,(64,171)).reshape(64,171)
input_1 = np.random.randint(0,2,(64, 171)).reshape(64,171)
input_2 = np.random.randint(0,2,(64, 171)).reshape(64,171)
softmax_min = np.array([0])
softmax_max = np.array([1])

model = Model(args.config, args.weight)
# for fp32
out = model.forward([input_0, input_1, input_2])
# for int8
# output = model.forward([input_0, input_1, input_2, softmax_min, softmax_max])
output = output[0].reshape(64,171,2)

print('input value is')
print(input_0)
print(input_1)
print(input_2)

print('output value is')
print(output)
