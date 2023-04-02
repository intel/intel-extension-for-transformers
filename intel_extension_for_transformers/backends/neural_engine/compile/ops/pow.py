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

# Given a tensor x and a tensor y ,
# this operation computes x^y for corresponding elements in x and y
@operator_registry(operator_type='Pow')
class Pow(Operator):
    """Register the Pow operator."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()
    def set_attr(self, framework, node):
        """Extract the node attr from onnxruntime."""
        if framework == 'torch':
            import numpy as np
            if node.inputsAt(1).type().kind() != 'TensorType':
                # import pdb; pdb.set_trace()
                data = np.array(node.inputsAt(1).toIValue()).astype(np.float32)
                input_tensor = Tensor(name=self.name + '_other',
                    source_op=[],
                    dest_op=[self.name],
                    shape=[1],
                    data=data,
                    dtype="fp32"
                    )
                self.input_tensors.append(input_tensor)