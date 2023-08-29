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
import copy

@operator_registry(operator_type='View')
class View(Operator):
    """Register the View operator."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from onnxruntime."""
        if framework == 'torch':
            shape_list = []
            if node.inputsAt(1).type().kind() == 'ListType':
                shape_list = parseTorchListConstruct(node.inputsAt(1))
            else:
                for i in range(1, node.inputsSize()):
                    shape_list.append(node.inputsAt(i).toIValue())
            shape_list = [-1 if x is None else x for x in shape_list]
            self._attr['shape'] = list2str(shape_list)