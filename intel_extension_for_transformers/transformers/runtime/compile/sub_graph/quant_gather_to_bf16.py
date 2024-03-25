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
"""The TorchInsertBF16Node Pattern."""

from .pattern import Pattern, pattern_registry
from collections import namedtuple, OrderedDict
from ..ops import Tensor
import numpy as np
from .. import graph_utils as util
from .. import logger


@pattern_registry(pattern_type='QuantGatherToBF16')
class TorchInsertBF16Node(Pattern):
    """The QuantGatherToBF16 pattern.

    Fuse the original sub-graph into the custom acceleration 'QuantGatherToBF16' graph.
    The search strategy is based on the following pattern mapping configs for different models.
    """

    def __call__(self, model):
        """The __call__ function of this pattern class."""

        def fp32_to_bf16(fp32_np):
            if fp32_np.dtype == np.float32:
                int32_np = fp32_np.view(dtype=np.int32)
                int32_np = int32_np >> 16
                bf16_np = int32_np.astype(np.uint16)
                return bf16_np
            elif fp32_np.dtype in [np.int16, np.uint16]:
                return fp32_np
            else:
                logger.error('Wrong dtype when convert to bf16 dtype.')

        if model.framework_modeling_config['framework'] != 'torch' or \
           util.get_autocast_info()['cast_type'] != "bf16":
            return model

        for node in model.nodes:
            if node.op_type == 'Gather' and node.attr and node.attr.get('embedding', False):
                weight_fp32 = node.input_tensors[1].data
                if isinstance(weight_fp32, np.ndarray):
                    node.input_tensors[1].data = fp32_to_bf16(weight_fp32)
                    node.input_tensors[1].dtype = 'bf16'
                    node.attr['output_dtype'] = 'bf16'

        return model
