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

"""The neural engine operator file."""

from abc import abstractmethod
from collections import namedtuple, OrderedDict
from .tensor import Tensor
from .. import logger
from ..graph_utils import list2str

OPERATORS = {}


def operator_registry(operator_type):
    """The class decorator used to register all Algorithm subclasses.

    Args:
        cls (class): The class of register.
        operator_type (str or str list): The operator registration name(s)

    Returns:
        cls: The class of register.
    """
    def decorator_operator(cls):
        if isinstance(operator_type, str):
            type_list = [operator_type]
        else:
            if not isinstance(operator_type, list):
                raise TypeError("Wrong input args, should be a string or a string list")
            type_list = operator_type
        for type in type_list:
            if type in OPERATORS:
                raise ValueError('Cannot have two operators with the same name')
            OPERATORS[type] = cls
        return cls

    return decorator_operator

def parseTorchListConstruct(lc_value):
    node = lc_value.node()
    values = []
    for i in range(node.inputsSize()):
        in_val = node.inputsAt(i)
        values.append(in_val.toIValue())
    return values

class Operator(object):
    """The class of neural engine operator."""

    def __init__(self):
        """The init function of this operator."""
        self._name = ''
        self._op_type = ''
        self._input_tensors= []
        self._output_tensors= []
        self._attr = OrderedDict()
        # ['extract_from_framework', 'construct']
        self._filling_method = None

    @property
    def name(self):
        """Get the operator name."""
        return self._name

    @name.setter
    def name(self, name):
        """Name assignment."""
        self._name = name

    @property
    def op_type(self):
        """Get op_type."""
        return self._op_type

    @op_type.setter
    def op_type(self, op_type):
        """Op_type assignment."""
        self._op_type = op_type

    @property
    def input_tensors(self):
        """Get input_tensors."""
        return self._input_tensors

    @input_tensors.setter
    def input_tensors(self, input_tensors):
        """Input tensor assignment."""
        self._input_tensors = input_tensors

    @property
    def output_tensors(self):
        """Get output tensor."""
        return self._output_tensors

    @output_tensors.setter
    def output_tensors(self, output_tensors):
        """Output_tensor assignment."""
        self._output_tensors = output_tensors

    @property
    def attr(self):
        """Get attr."""
        return self._attr

    @attr.setter
    def attr(self, attr):
        """Attr assignment."""
        self._attr = attr

    def set_attr(self, framework, node):
        """Attr initialization."""
        self._attr = OrderedDict()

    @property
    def filling_method(self):
        return self._filling_method

    def extract(self, framework, node, framework_model, nodes_dict, engine_graph=None):
        """Extract the op from framework."""
        from ..tf_utils import tf_extract_operator
        from ..onnx_utils import onnx_extract_operator
        from ..torch_utils import torch_extract_operator

        OP_EXTRACTORS = {
            'tensorflow': tf_extract_operator,
            'onnxruntime': onnx_extract_operator,
            'torch': torch_extract_operator,
        }
        if framework == "torch":
            from ..torch_utils import get_node_name
            self._name = get_node_name(node)
        else:
            self._name = node.name
        self._op_type, self._input_tensors, self._output_tensors = OP_EXTRACTORS[framework](
            node, framework_model, nodes_dict, engine_graph)
        self.set_attr(framework, node)
        self._filling_method = 'extract_from_' + framework

    def construct(self, name, op_type, input_tensors=[], output_tensors=[], attr=OrderedDict()):
        """Make the op by set the attributes."""
        self._name = name
        self._op_type = op_type
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._attr = attr
        self._filling_method = 'construct'

    @property
    def config(self):
        """Get the op config in the graph."""
        conf_dict = OrderedDict()
        # conf_dict['type'] = self._op_type
        conf_dict['type'] = self._op_type
        if len(self._input_tensors) > 0:
            conf_dict['input'] = OrderedDict()
            for input_tensor in self._input_tensors:
                if self._op_type == 'Input':
                    conf_dict['input'][input_tensor.name] = input_tensor.config
                else:
                    conf_dict['input'][input_tensor.name] = {}
        if len(self._output_tensors) > 0:
            conf_dict['output'] = OrderedDict()
            for output_tensor in self._output_tensors:
                if self._op_type == 'Input':
                    conf_dict['output'][output_tensor.name] = output_tensor.config
                else:
                    conf_dict['output'][output_tensor.name] = {}

        if isinstance(self._attr, dict) and len(self._attr.keys()) > 0:
            conf_dict['attr'] = self._attr

        return conf_dict
