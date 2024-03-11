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
import copy

# tf.shape(input, out_type=tf.dtypes.int32, name=None)
# Returns a tensor containing the shape of the input tensor.
@operator_registry(operator_type='Shape')
class Shape(Operator):
    """Register the Shape operator."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()
    def set_attr(self, framework, node):
        """Extract the node attr."""
        if framework == 'torch':
            self._attr['start'] = node.inputsAt(1).toIValue()
            self._attr['end'] = node.inputsAt(1).toIValue() + 1
