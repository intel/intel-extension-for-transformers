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

from .op import Operator, operator_registry, parseTorchListConstruct
from .tensor import Tensor
from ..graph_utils import list2str
import copy


# tf.transpose(a, perm=None, conjugate=False, name='transpose')
# The returned tensor's dimension i will correspond to the input dimension perm[i].
# If perm is not given, it is set to (n-1...0), where n is the rank of the input tensor.
# Hence by default, this operation performs a regular matrix transpose on 2-D input Tensors.
# If conjugate is True and a.dtype is either complex64 or complex128 then the values of a are
# conjugated and transposed.
# normally, we don't need the param 'conjugate'
@operator_registry(operator_type='Transpose')
class Transpose(Operator):
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
                self._attr['src_perm'] = list2str(list(range(node.inputsAt(1).node().inputsSize())))
                self._attr['dst_perm'] = list2str(parseTorchListConstruct(node.inputsAt(1)))
            if node.kind() == 'aten::transpose':
                dim0 = node.inputsAt(1).toIValue()
                dim1 = node.inputsAt(2).toIValue()
                self._attr['transpose_dims'] = list2str([dim0, dim1])