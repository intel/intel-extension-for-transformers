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


# tf.nn.gelu(features, approximate=False, name=None)
# approximate=True: return x * cdf
# cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
# approximate=False: return x * 0.5 * (1.0 + math_ops.erf(x / math.sqrt(2.0)))
@operator_registry(operator_type='Gelu')
class Gelu(Operator):
    """Parse the Gelu operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def set_attr(self, framework, node):
        """Extract the node attr from tensorflow."""
        if framework == 'tensorflow':
            self._attr['approximate'] = node.attr['approximate'].b
        if framework == 'torch':
            approximate = node.inputsAt(1).toIValue()
            if approximate == 'none':
                self._attr['algorithm'] = 'gelu_erf'
            elif approximate == 'tanh':
                self._attr['algorithm'] = 'gelu_tanh'
