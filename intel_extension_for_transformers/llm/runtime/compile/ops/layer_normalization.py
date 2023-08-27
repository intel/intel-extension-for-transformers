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

"""The neural engine operator mapping file."""

from .op import Operator, operator_registry, list2str, parseTorchListConstruct
from .tensor import Tensor


@operator_registry(operator_type='LayerNormalization')
class LayerNormalization(Operator):
    """Parse the LayerNormalization operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from frameworks."""
        self._op_type = 'LayerNorm'
        if framework == "onnxruntime":
            axis = node.attribute[1].i
            if axis != -1:
                self._attr['axis'] = axis
            self._attr['epsilon'] = node.attribute[2].f

# Fused_op Mean, AddV2, Mul, etc.
# This pattern has several ops combinations, so the input_tensors and output_tensors may various
@operator_registry(operator_type='LayerNorm')
class LayerNorm(Operator):
    """Register the LayerNorm operator."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        if framework == 'torch':
            """Extract the node attr from torchscript."""
            if node.inputsSize() > 4:
                self._attr['epsilon'] = node.inputsAt(4).toIValue()
                self._attr['normalized_shape'] = list2str(parseTorchListConstruct(node.inputsAt(1)))