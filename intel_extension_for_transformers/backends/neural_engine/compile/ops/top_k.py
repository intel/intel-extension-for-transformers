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

from .op import Operator, operator_registry
from .tensor import Tensor
from ..graph_utils import list2str


@operator_registry(operator_type='TopK')
class TopK(Operator):
    """Parse the TopK operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from onnxruntime."""
        if framework == 'onnxruntime':
            for attribute in node.attribute:
                if attribute.name == 'axis':
                    self._attr['axis'] = attribute.i
                if attribute.name == 'largest':
                    self._attr['largest'] = attribute.i
                if attribute.name == 'sorted':
                    self._attr['sorted'] = attribute.i
            # write the K value into attr if it's const
            if self._input_tensors[1].data:
                self._attr['k'] = int(self._input_tensors[1].data)
                self._input_tensors.pop()
            # remove the data output_tensor, just keep indices
            self._output_tensors = [self._output_tensors[1]]