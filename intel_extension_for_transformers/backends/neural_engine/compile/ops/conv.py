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

"""The neural engine operator mapping file."""

from .op import Operator, operator_registry
from .tensor import Tensor
from ..graph_utils import list2str

def parseTorchListConstruct(lc_value):
    node = lc_value.node()
    values = []
    for i in range(node.inputsSize()):
        in_val = node.inputsAt(i)
        values.append(in_val.toIValue())
    return values

def parseTorchConstant(lc_value):
    node = lc_value.node()
    values = []
    for i in range(node.inputsSize()):
        in_val = node.inputsAt(i)
        values.append(in_val.toIValue())
    return values

@operator_registry(operator_type='Conv')
class Conv(Operator):
    """Parse the Conv operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from onnxruntime."""
        if framework == 'onnxruntime':
            for attribute in node.attribute:
                if attribute.name == 'dilations':
                    self._attr['dilations'] = list2str(attribute.ints)
                if attribute.name == 'group':
                    self._attr['group'] = int(attribute.i)
                if attribute.name == 'kernel_shape':
                    self._attr['kernel_shape'] = list2str(attribute.ints)
                if attribute.name == 'pads':
                    self._attr['pads'] = list2str(attribute.ints)
                if attribute.name == 'strides':
                    self._attr['strides'] = list2str(attribute.ints)
        elif framework == "torch":
            assert node.inputsSize() == 12
            self._attr['strides'] = list2str(parseTorchListConstruct(node.inputsAt(3)))
            self._attr['pads'] = list2str(parseTorchListConstruct(node.inputsAt(4)))
            self._attr['dilations'] = list2str(parseTorchListConstruct(node.inputsAt(5)))
            self._attr['transposed'] = node.inputsAt(6).toIValue()
            self._attr['output_padding'] = list2str(parseTorchListConstruct(node.inputsAt(7)))
            self._attr['group'] = node.inputsAt(8).toIValue()
            self._attr['benchmark'] = node.inputsAt(9).toIValue()
            self._attr['deterministic'] = node.inputsAt(10).toIValue()
            self._attr['cudnn_enabled'] = node.inputsAt(11).toIValue()

