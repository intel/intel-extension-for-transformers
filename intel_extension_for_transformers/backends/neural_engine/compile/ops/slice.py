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
from ..graph_utils import list2str
import numpy as np


@operator_registry(operator_type='Slice')
class Slice(Operator):
    def __init__(self):
        super().__init__()

    def set_attr(self, framework, node):
        if framework == 'onnxruntime':
            keep_idx = [0]
            # axes and steps are optional input tensors
            name_idx = {1: "starts", 2: "ends", 3: "axes", 4: "steps"}
            assert len(self._input_tensors) > 2, \
                "wrong input_tensors in Slice Op, should have 3 input tensors at least."
            for idx, input_tensor in enumerate(self._input_tensors):
                if idx == 0:
                    continue
                if isinstance(input_tensor.data, np.ndarray):
                    self._attr[name_idx[idx]] = list2str(list(input_tensor.data))
                else:
                    keep_idx.append(idx)
            self._input_tensors = [self._input_tensors[i] for i in keep_idx]

        if framework == 'torch':
            if node.kind() == 'aten::slice':
                self._attr['axes'] = node.inputsAt(1).toIValue()
                self._attr['starts'] = node.inputsAt(2).toIValue()
                self._attr['ends'] = node.inputsAt(3).toIValue()   # TODO: check None type - 1
                self._attr['steps'] = node.inputsAt(4).toIValue()
            elif node.kind() == 'aten::select':
                self._attr['axes'] = node.inputsAt(1).toIValue()
                self._attr['starts'] = node.inputsAt(2).toIValue()
                self._attr['ends'] = node.inputsAt(2).toIValue()
                self._attr['steps'] = 1
