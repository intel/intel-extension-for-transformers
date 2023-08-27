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


# tf.placeholder(dtype, shape=None, name=None)
@operator_registry(operator_type='IteratorGetNext')
class IteratorGetNext(Operator):
    """Parse the IteratorGetNext operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from tensorflow."""
        if framework == 'tensorflow':
            from ..tf_utils import TF_DTYPE_ID
            output_shapes = []
            shape_list = node.attr['output_shapes'].list.shape
            for each in shape_list:
                tmp = []
                shape = each.dim
                for s in shape:
                    tmp.append(s.size)
                output_shapes.append(tmp)
            self._attr['output_shapes'] = output_shapes
            output_types = node.attr['output_types'].list.type
            output_types = [TF_DTYPE_ID[i] for i in output_types]
            self._attr['output_types'] = output_types
            for i in range(len(output_types)):
                self._output_tensors[i].dtype = output_types[i]
