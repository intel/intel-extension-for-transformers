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


# The inputs must be two-dimensional matrices
@operator_registry(operator_type='FusedGemm')
class FusedGemm(Operator):
    """Parse the FusedGemm operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from frameworks."""
        activation_dict = {'Tanh': 'tanh'}
        self._op_type = 'InnerProduct'
        if framework == 'onnxruntime':
            activation = None
            transpose_a = False
            transpose_b = False
            alpha = 1.0
            beta = 1.0
            for attr in node.attribute:
                if attr.name == 'activation':
                    activation = str(attr.s, encoding='utf-8')
                if attr.name == 'transA':
                    transpose_a = bool(attr.i)
                if attr.name == 'transB':
                    transpose_b = bool(attr.i)
                if attr.name == 'alpha':
                    alpha = attr.f
                if attr.name == 'beta':
                    beta = attr.f
            if transpose_a:
                self._attr['src0_perm'] = '1,0'
            # see OneDNN InnerProduct related requirements
            if not transpose_b:
                self._attr['src1_perm'] = '1,0'
            else:
                self._attr['src1_perm'] = '0,1'
            if alpha != 1.0:
                self._attr['alpha'] = alpha
            if beta != 1.0:
                self._attr['beta'] = beta

            self._attr['append_op'] = activation_dict[activation]
