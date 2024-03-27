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
from ..graph_utils import names_from_input
from .. import logger


# graph.input
@operator_registry(operator_type='ONNXINPUT')
class ONNXINPUT(Operator):
    """Parse the ONNXINPUT operator to the neural engine."""
    def __init__(self):
        """The init function of this operator."""
        super().__init__()

    def extract(self, framework, node, framework_model, nodes_dict, engine_graph=None):
        """Extract operators to the neural engine."""
        from ..onnx_utils import ONNX_DTYPE_ID
        self._name = node.name
        self._op_type = 'ONNXINPUT'
        output_tensor_name = names_from_input(self._name)[1]
        shape_len = len(node.type.tensor_type.shape.dim)
        shape = [-1] * shape_len
        for i, d in enumerate(node.type.tensor_type.shape.dim):
            param = d.dim_param
            if engine_graph and param in ['max_seq_len', 'seq_len'] and i == 0:
                engine_graph.add_config_item('seq_len_first_dim', True)
            if param == '':
                    v = d.dim_value
                    if v != 0:
                        shape[i] = v
                    else:
                        logger.error("Unknown dimension parameter in ONNX model input, " \
                                 "only dim_param or dim_value")
        dtype = ONNX_DTYPE_ID[node.type.tensor_type.elem_type]
        output_tensor = Tensor(
            name=output_tensor_name,
            shape=shape,
            dtype=dtype,
            source_op=[self._name],
            dest_op=nodes_dict[self._name].outputs,
        )

        self._output_tensors = [output_tensor]
