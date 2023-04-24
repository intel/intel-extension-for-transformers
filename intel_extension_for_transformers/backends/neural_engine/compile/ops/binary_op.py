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

from .op import Operator, operator_registry
from .tensor import Tensor
import numpy as np
from .. import logger


@operator_registry(operator_type=['Div', 'Sub', 'Mul', 'Equal', 'Less', 'Greater',
                                  'LessEqual', 'LessOrEqual', 'GreaterOrEqual',
                                  'NonZero', 'Not', 'Neg', 'NotEqual'])
class BinaryOp(Operator):
    def __init__(self):
        super().__init__()
        self._algorithm_dict = {'Addv2': 'add', 'Add': 'add', 'Div': 'div', 'Sub': 'sub',
                                'Mul': 'mul', 'Equal': 'eq', 'Less': 'lt', 'Greater': 'gt',
                                'LessEqual': 'le', 'LessOrEqual': 'le', 'GreaterOrEqual': 'ge',
                                'NonZero': 'ne', 'NotEqual': 'ne', 'Not': 'eq', 'Neg': 'mul'}

    def set_attr(self, framework, node):
        if framework == "onnxruntime":
            self._attr['algorithm'] = self._algorithm_dict[self._op_type]
            # deal with const input tensor
            if len(self._input_tensors) == 2:
                const_idx = -1
                for i in [0, 1]:
                    if isinstance(self._input_tensors[i].data, np.ndarray):
                        const_idx = i
                        break
                if const_idx != -1:
                    if len(self._input_tensors[const_idx].data.shape) == 0:
                        self._input_tensors[const_idx].shape = [1]
                        self._input_tensors[const_idx].data = \
                            self._input_tensors[const_idx].data.reshape(1)
                    # dnnl Binary op only supports f32, bf16, f16, u8, s8
                    # convert dtype int64 and int32 into dtype fp32
                    const_data = self._input_tensors[const_idx].data
                    if const_data.dtype in [np.int64, np.int32]:
                        if all(const_data >= np.finfo(np.float32).min) and \
                           all(const_data <= np.finfo(np.float64).max):
                            self._input_tensors[const_idx].data = const_data.astype(np.float32)
                        else:
                            logger.warning("Cannot convert const data into fp32 dtype in op type"\
                                          " {}".format(self._op_type))

            # convert other special ops to binary ops
            if self._op_type in ["NonZero", "Not"]:
                self._input_tensors.append(Tensor(name=self._name + "_equal_val",
                                                  data=np.array([0]).astype(np.float32),
                                                  shape=[1],
                                                  dest_op=[self._name])
                                           )
            if self._op_type in ["Neg"]:
                self._input_tensors.append(Tensor(name=self._name + "_mul_val",
                                                  data=np.array([-1]).astype(np.float32),
                                                  shape=[1],
                                                  dest_op=[self._name])
                                           )
        if framework == 'torch':
            algo_dict = {'aten::rsub': 'sub', 'aten::mul': 'mul', 'aten::add': 'add', 'aten::add_': 'add',
                         'aten::div': 'div', 'aten::sub': 'sub', 'aten::gt': 'gt', 'aten::lt': 'lt',
                         'aten::eq': 'eq', 'aten::ne': 'ne', 'aten::neg': 'mul',
                         'aten::floor_divide': 'div'}
            self._attr['algorithm'] = algo_dict[node.kind()]
            if node.kind() == 'aten::neg':
                self._input_tensors.append(Tensor(name=self._name + "_mul_val",
                                                  data=np.array([-1]).astype(np.float32),
                                                  shape=[1],
                                                  dest_op=[self._name]))