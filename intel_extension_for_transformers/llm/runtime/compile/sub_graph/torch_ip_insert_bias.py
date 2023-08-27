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

"""The TorchInnerProductInsertBias Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from .. import graph_utils as util
import copy
from ..ops import Tensor
import numpy as np

@pattern_registry(pattern_type='TorchInnerProductInsertBias')
class TorchInnerProductInsertBias(Pattern):
    """The TorchInnerProductInsertBias pattern.

    """
    def __call__(self, model):
        """The __call__ function of this pattern class."""
        if model.framework_modeling_config['framework'] != 'torch' or not util.get_quant_info():
            return model
        for node in model.nodes:
            if node.op_type == 'InnerProduct':
                if len(node.input_tensors) == 2:
                    # insert bias
                    weight = node.input_tensors[1].data
                    bias = Tensor(name = node.name + '_bias',
                                source_op = [],
                                dest_op = [node.name],
                                shape = [weight.shape[0]],
                                data = np.zeros(weight.shape[0]).astype(np.float32),
                                dtype = 'fp32'
                                )
                    node.input_tensors.append(bias)
        return model
