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

from .op import Operator, operator_registry, list2str
from .tensor import Tensor
import copy

@operator_registry(operator_type='Reorder')
class Reorder(Operator):
    """Parse the Transpose operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from onnxruntime."""
        if framework == "tensorflow":
            if self._input_tensors[1].source_op == []:
                dst_perm = list(self._input_tensors[1].data)
                self._attr['dst_perm'] = list2str(dst_perm)
                src_perm = [i for i in range(len(dst_perm))]
                self._attr['src_perm'] = list2str(src_perm)
                _ = self._input_tensors.pop()

        if framework == "onnxruntime":
            dst_perm = node.attribute[0].ints
            src_perm = [i for i in range(len(dst_perm))]
            self._attr['src_perm'] = list2str(src_perm)
            self._attr['dst_perm'] = list2str(dst_perm)

        if framework == 'torch':
            if node.kind() == 'aten::permute':
                # import pdb; pdb.set_trace()
                self._attr['src_perm'] = list2str(list(range(len(node.inputsAt(1).toIValue()))))
                self._attr['dst_perm'] = list2str(node.inputsAt(1).toIValue())
            if node.kind() == 'aten::transpose':
                dim0 = node.inputsAt(1).toIValue()
                dim1 = node.inputsAt(2).toIValue()
                self._attr['transpose_dims'] = list2str([dim0, dim1])
