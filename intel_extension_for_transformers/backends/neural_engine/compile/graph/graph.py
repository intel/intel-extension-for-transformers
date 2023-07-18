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
"""The neural engine graph file."""

import re
from ..ops import Operator
from collections import OrderedDict, namedtuple
from .. import logger
import numpy as np
import yaml
import os
import copy
import time


class Graph(object):
    """The defintion of the neural engine graph."""

    def __init__(self):
        """The graph initialization."""
        self._nodes = []
        self._node_id = {}
        self._engine = None
        self._execution_options = None
        self._refresh_execution_options = False
        # modeling config, like num_attention_heads in transformer related models
        self._framework_modeling_config = {'framework': None}
        self._input_tensors_name = []
        self._output_tensors_name = []
        self._max_input_shapes_list = None
        self._do_activation_mem_compression = True if "ENGINE_ACTIVATION_MEM_COMPRESSION" in \
                                              os.environ else False
        self._refresh_max_input_shapes_list = False
        # For engine_init.
        # Only init them when load_weight = False
        self._config_root = None
        self._weight_root = None

    @property
    def nodes(self):
        """Get nodes."""
        return self._nodes

    @nodes.setter
    def nodes(self, new_nodes):
        """Nodes assignment."""
        self._nodes = new_nodes

    @property
    def execution_options(self):
        """The options for execution."""
        logger.warning(
            "execution_options' option values are constants. "
            "Please reset the execution_option property if you want change some "
            "options when inference, like 'graph.execution_options = your_new_options'. "
            "Do not use 'graph.execution_options.some_option = value' directly!")
        # pylint: disable=E0611
        import intel_extension_for_transformers.neural_engine_py as ne
        options = ne.ExecutionOptions()
        options_list = [
            option for option in dir(options)
            if not option.startswith("__") and not option.startswith("__")
        ]
        execution_option_ret = namedtuple("ExecutionOptions", options_list)
        values = [None] * len(options_list)
        if self._execution_options:
            values = [getattr(self._execution_options, option) for option in options_list]
        return execution_option_ret._make(values)

    @execution_options.setter
    def execution_options(self, options):
        """Set execution_options"""
        # pylint: disable=E0611
        import intel_extension_for_transformers.neural_engine_py as ne
        if isinstance(options, ne.ExecutionOptions):
            self._execution_options = options
        else:
            assert isinstance(options, dict), "options should a dictionary!"
            ne_options = ne.ExecutionOptions()
            # convert user's dict
            def set_execution_mode(v):
                assert isinstance(v, str), "execution_mode should be string."
                if v == "inference":
                    ne_options.execution_mode = ne.ExecutionMode.INFERENCE
                elif v == "debug":
                    ne_options.execution_mode = ne.ExecutionMode.DEBUG
                elif v == "tuning":
                    ne_options.execution_mode = ne.ExecutionMode.TUNING
                else:
                    logger.error('execution_mode value should be inference, debug or tuning.')

            def set_enable_op_tuning(v):
                assert isinstance(v, bool), "enable_op_tuning shoule be True or False."
                ne_options.enable_op_tuning = v

            def set_warmup_iter(v):
                assert isinstance(v, int), "warmup_iter should be int."
                ne_options.warmup_iter = v

            def set_dispatch_table_file_root(v):
                assert isinstance(v, str), "dispatch_table_file_root should be string."
                ne_options.dispatch_table_file_root = os.path.join(os.getcwd(), v)

            def set_activation_mem_compression(v):
                assert isinstance(v, bool), "activation_mem_compression shoule be True or False."
                ne_options.activation_mem_compression = v

            def set_dump_activation_dag(v):
                assert isinstance(v, bool), "dump_activation_dag shoule be True or False."
                ne_options.dump_activation_dag = v

            convert_options = {'execution_mode': set_execution_mode,
                               'enable_op_tuning': set_enable_op_tuning,
                               'warmup_iter': set_warmup_iter,
                               'dispatch_table_file_root': set_dispatch_table_file_root,
                               'activation_mem_compression': set_activation_mem_compression,
                               'dump_activation_dag': set_dump_activation_dag}
            for k, v in options.items():
                if k in convert_options:
                    convert_options[k](v)
                else:
                    logger.error("{} is invalid execution options.".format(k))
            self._execution_options = ne_options

        self._refresh_execution_options = True
        self._do_activation_mem_compression = self._execution_options.activation_mem_compression

    @property
    def framework_modeling_config(self):
        """Get framework_modeling_config"""
        return self._framework_modeling_config

    @framework_modeling_config.setter
    def framework_modeling_config(self, config):
        """Set framework_modeling_config"""
        self._framework_modeling_config = config

    def add_config_item(self, key, val):
        """Add a pair into framework_modeling_config"""
        self._framework_modeling_config[key] = val

    def inquire_config_item(self, key):
        """Get a pair from framework_modeling_config"""
        val = self._framework_modeling_config.get(key, None)
        if not val:
            logger.warning("the item {} does not exist in the config...".format(key))
        return val

    @property
    def input_tensors_name(self):
        """Get input_tensors_name"""
        return self._input_tensors_name

    @input_tensors_name.setter
    def input_tensors_name(self, name_list):
        """Set input_tensors_name"""
        self._input_tensors_name = name_list

    @property
    def output_tensors_name(self):
        """Get output_tensors_name"""
        return self._output_tensors_name

    @output_tensors_name.setter
    def output_tensors_name(self, name_list):
        """Set output_tensors_name"""
        self._output_tensors_name = name_list

    @property
    def max_input_shapes_list(self):
        """Get max_input_shapes"""
        return self._max_input_shapes_list

    @max_input_shapes_list.setter
    def max_input_shapes_list(self, input_shapes_list):
        """Set max_input_shapes"""
        assert (isinstance(input_shapes_list, list) and isinstance(input_shapes_list[0], list) \
               and isinstance(input_shapes_list[0][0], list)) or input_shapes_list == None, \
               "max_input_shape should be None or List[List[List]]!"
        self._max_input_shapes_list = input_shapes_list
        self._refresh_max_input_shapes_list = True

    def insert_nodes(self, index, nodes):
        """Insert nodes to neural engine IR."""
        idx = index
        for node in nodes:
            node = self.modify_node_connections(node, mode='insert')
            self._nodes.insert(idx, node)
            self._node_id[node.name] = idx
            for i in range(idx + 1, len(self._nodes)):
                self._node_id[self._nodes[i].name] += 1
            idx += 1
        self._engine = None

    def remove_nodes(self, node_names):
        """Remove nodes from neural engine IR."""
        for node_name in node_names:
            if node_name not in self._node_id.keys():
                continue
            node = self.get_node_by_name(node_name)
            _ = self.modify_node_connections(node, mode='remove')
            index = self.get_node_id(node_name)
            for i in range(index + 1, len(self._nodes)):
                self._node_id[self._nodes[i].name] -= 1
            self._nodes.pop(index)
            self._node_id.pop(node_name)
        self._engine = None

    def get_node_id(self, node_name):
        """Get the node id according to the node name."""
        try:
            index = self._node_id[node_name]
            return index
        except BaseException:
            raise ValueError(
                'There is no node named {}, please check the input name.'.format(node_name))

    def get_node_by_name(self, node_name):
        """Get the node according to the node name."""
        index = self.get_node_id(node_name)
        if index is not None:
            return self._nodes[index]
        else:
            return None

    def rename_node(self, old_name, new_name):
        """Modify the node name."""
        index = self.get_node_id(old_name)
        for i in range(len(self._nodes[index].input_tensors)):
            self._nodes[index].input_tensors[i].dest_op = [new_name]
            for pre_node_name in self._nodes[index].input_tensors[i].source_op:
                tensor_idx = self.get_tensor_idx(pre_node_name,
                                                 self._nodes[index].input_tensors[i].name)
                pre_node_idx = self._node_id[pre_node_name]
                self._nodes[pre_node_idx].output_tensors[tensor_idx].dest_op.remove(old_name)
                self._nodes[pre_node_idx].output_tensors[tensor_idx].dest_op.append(new_name)
        for i in range(len(self._nodes[index].output_tensors)):
            self._nodes[index].output_tensors[i].source_op = [new_name]
            for next_node_name in self._nodes[index].output_tensors[i].dest_op:
                tensor_idx = self.get_tensor_idx(next_node_name,
                                                 self._nodes[index].output_tensors[i].name,
                                                 from_output=False)
                next_node_idx = self._node_id[next_node_name]
                self._nodes[next_node_idx].input_tensors[tensor_idx].source_op = [new_name]

        self._nodes[index].name = new_name
        self._node_id.pop(old_name)
        self._node_id[new_name] = index
        self._engine = None

    def change_node_input_tensors(self, node_name, index, tensor=None, mode='modify'):
        """Modify the input tensors of the node."""
        assert mode in ['insert', 'remove', 'modify'], 'Wrong mode'
        node = self.get_node_by_name(node_name)
        index = index if index >= 0 else len(node.input_tensors) + index
        node_index = self.get_node_id(node_name)
        source_node_idx = None
        tensor_idx = None
        if mode == 'remove':
            tensor = node.input_tensors[index]
        assert tensor is not None
        tensor.dest_op = [node_name]
        if tensor.source_op != []:
            source_node_idx = self.get_node_id(tensor.source_op[0])
            tensor_idx = self.get_tensor_idx(tensor.source_op[0], tensor.name, from_output=True)

        if mode == 'insert':
            if source_node_idx is not None:
                if node_name not in \
                        self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op:
                    self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.append(
                        node_name)
            self._nodes[node_index].input_tensors.insert(index, tensor)
        elif mode == 'remove':
            if source_node_idx is not None:
                self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.remove(node_name)
            self._nodes[node_index].input_tensors.pop(index)

        else:
            self.change_node_input_tensors(node_name, index, mode='remove')
            self.change_node_input_tensors(node_name, index, tensor=tensor, mode='insert')
        self._engine = None

    def change_node_output_tensors(self, node_name, index, tensor=None, mode='modify'):
        """Modify the output tensors of the node."""
        assert mode in ['insert', 'remove', 'modify'], 'Wrong mode'
        node = self.get_node_by_name(node_name)
        index = index if index >= 0 else len(node.output_tensors) + index
        node_index = self.get_node_id(node_name)
        if mode == 'remove':
            tensor = node.output_tensors[index]
        assert tensor is not None
        tensor.source_op = [node_name]

        if mode == 'insert':
            self._nodes[node_index].output_tensors.insert(index, tensor)
        elif mode == 'remove':
            self._nodes[node_index].output_tensors.pop(index)
        else:
            self._nodes[node_index].output_tensors[index] = tensor
        self._engine = None

    def get_pre_node_names(self, node_name):
        """Get the previous node name."""
        pre_node_names = []
        node = self.get_node_by_name(node_name)
        for input_tensor in node.input_tensors:
            if input_tensor.source_op != []:
                pre_node_names.extend(input_tensor.source_op)

        return pre_node_names

    def get_next_node_names(self, node_name):
        """Get the next node name."""
        next_node_names = []
        node = self.get_node_by_name(node_name)
        for output_tensor in node.output_tensors:
            if output_tensor.dest_op != []:
                next_node_names.extend(output_tensor.dest_op)

        return next_node_names

    def get_tensor_idx(self, node_name, tensor_name, from_output=True):
        """Get the index of tensor."""
        target_node = self.get_node_by_name(node_name)
        tensor_idx = -1
        if from_output:
            target_tensors = target_node.output_tensors
        else:
            target_tensors = target_node.input_tensors
        for j in range(len(target_tensors)):
            if target_tensors[j].name == tensor_name:
                tensor_idx = j
                break
            else:
                continue
        # assert tensor_idx != -1, 'Graph does not has tensor {}, '\
        #'please check it.'.format(tensor_name)

        return tensor_idx

    def modify_node_connections(self, node, mode='insert'):
        """Modify the order of nodes."""
        assert mode in ['insert', 'remove'], 'Wrong mode {}'.format(mode)
        # modify the input_tensors' source_op
        for i in range(len(node.input_tensors)):
            node.input_tensors[i].dest_op = [node.name]
            t = node.input_tensors[i]
            if t.source_op != [] and t.source_op[0] in self._node_id.keys():
                source_node_idx = self.get_node_id(t.source_op[0])
                source_node = self._nodes[source_node_idx]
                tensor_idx = self.get_tensor_idx(source_node.name, t.name)
                if mode == 'insert':
                    if node.name not in \
                            self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op:
                        self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.append(
                            node.name)
                if mode == 'remove':
                    if node.name in \
                            self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op:
                        self._nodes[source_node_idx].output_tensors[tensor_idx].dest_op.remove(
                            node.name)
            # skip the const tensor and the node has been removed
            else:
                continue

        # modify the output_tensors' dest_op
        for i in range(len(node.output_tensors)):
            t = node.output_tensors[i]
            for dest_op_name in node.output_tensors[i].dest_op:
                if dest_op_name in self._node_id.keys():
                    dest_node_idx = self.get_node_id(dest_op_name)
                    tensor_idx = self.get_tensor_idx(dest_op_name, t.name, from_output=False)
                    if mode == 'insert':
                        node.output_tensors[i].source_op = [node.name]
                        if tensor_idx != -1:
                            self._nodes[dest_node_idx].input_tensors[tensor_idx].source_op = [
                                node.name
                            ]
                    if mode == 'remove':
                        if tensor_idx != -1 and node.name not in self._node_id.keys():
                            if node.name in \
                            self._nodes[dest_node_idx].input_tensors[tensor_idx].source_op:
                                self._nodes[dest_node_idx].input_tensors[tensor_idx].\
                                source_op.remove(node.name)
        self._engine = None

        return node

    @property
    def weight_data(self):
        """Get the weight_bytes to bin file."""
        consts_info = OrderedDict()
        weight_bytes = bytearray()
        non_consts_len = 0

        for t in self._nodes[0].output_tensors:
            assert self._nodes[0].op_type == 'Input', 'The graph must have input data'
            if t.source_op == [] and isinstance(t.data, np.ndarray):
                break
            else:
                non_consts_len += 1
        self._nodes[0].output_tensors = self._nodes[0].output_tensors[:non_consts_len]
        const_idx = non_consts_len - 1
        const_info = {}
        for i in range(len(self._nodes)):
            for j in range(len(self._nodes[i].input_tensors)):
                t = self._nodes[i].input_tensors[j]
                if (t.source_op == [] or self._node_id.get(t.source_op[-1], None) == 0) \
                                      and isinstance(t.data, np.ndarray):
                    if t.name not in const_info:
                        const_idx += 1
                        const_info[t.name] = const_idx
                        data = t.data
                        start = len(weight_bytes)
                        data_bytes = data.tobytes()
                        weight_bytes.extend(data_bytes)
                        offset = len(data_bytes)
                        self._nodes[i].input_tensors[j].location = [start, offset]
                        self._nodes[0].output_tensors.append(self._nodes[i].input_tensors[j])
                    else:
                        if self._nodes[0].output_tensors[const_info[t.name]].data.dtype != \
                           self._nodes[i].input_tensors[j].data.dtype:
                            logger.error("tensor {} has different dtypes in multi-nodes!".
                                         format(t.name))
                        dest_op = set(self._nodes[0].output_tensors[const_info[t.name]].dest_op +
                                      self._nodes[i].input_tensors[j].dest_op)
                        self._nodes[0].output_tensors[const_info[t.name]].dest_op = list(dest_op)
        weight_bytes = bytes(weight_bytes)
        return weight_bytes

    @property
    def net_config(self):
        """Get the network config dict to yaml file."""
        net_info = OrderedDict()
        net_info['model'] = OrderedDict()
        net_info['model']['name'] = 'model'
        net_info['model']['operator'] = OrderedDict()
        for node in self._nodes:
            net_info['model']['operator'][node.name] = node.config

        return net_info

    def dump_tensor(self, tensor_list=[]):
        """Dump tensors to the bin and yaml file."""
        weight_data = self.weight_data
        net_info = self.net_config
        if tensor_list == []:
            for node in net_info['model']['operator']:
                if 'output' in net_info['model']['operator'][node].keys():
                    for tensor in net_info['model']['operator'][node]['output']:
                        if 'location' in \
                                net_info['model']['operator'][node]['output'][tensor].keys():
                            continue
                        net_info['model']['operator']['output_data']['input'][tensor] = {}
        else:
            for tensor in tensor_list:
                for node in net_info['model']['operator']:
                    operator = net_info['model']['operator']
                    if 'output' not in operator[node].keys():
                        continue
                    for tensor_name in operator[node]['output']:
                        search = re.search(tensor, tensor_name, re.I)
                        if search is not None:
                            net_info['model']['operator']['output_data']['input'][tensor_name] = {}

        return net_info

    def engine_init(self, net_info={}, weight_data=b"", refresh_model=True):
        """Pybind engine executor."""
        if refresh_model:
            # pylint: disable=E0611
            import intel_extension_for_transformers.neural_engine_py as ne
            if not weight_data:
                weight_data = self._weight_root if self._weight_root else self.weight_data
            if not net_info:
                net_info = self._config_root if self._config_root else self.net_config
            op_configs = []
            tensor_output = []
            tensor_input = []
            attr_map_list = []
            output_list = []
            if isinstance(net_info, dict) and not self._config_root:
                for node in net_info['model']['operator']:
                    tensor_input.append([])
                    tensor_output.append([])
                    opeartor = net_info['model']['operator'][node]
                    if 'input' in opeartor.keys():
                        for input_name in opeartor['input']:
                            input_tensor = ne.tensor_config(input_name, [], "fp32", [], [])
                            tensor_input[-1].append(input_tensor)

                    if 'output' in opeartor.keys():
                        for (output_name, attrs) in opeartor['output'].items():
                            tensor_location = []
                            if 'location' in attrs.keys():
                                tensor_location = attrs['location']
                            tensor_strides = []
                            if "strides" in attrs.keys():
                                tensor_strides = attrs["strides"]
                            tensor_shape = []
                            if "shape" in attrs.keys():
                                tensor_shape = attrs["shape"]
                            tensor_dtype = 'fp32'
                            if "dtype" in attrs.keys():
                                tensor_dtype = attrs["dtype"]
                            output_tensor = ne.tensor_config(output_name, tensor_shape,
                                                             tensor_dtype, tensor_strides,
                                                             tensor_location)
                            tensor_output[-1].append(output_tensor)

                    if 'attr' in opeartor.keys():
                        op_attr = opeartor['attr']
                        attr_maps = {}
                        for (k, v) in op_attr.items():
                            attr_maps[str(k)] = str(v)
                        attr_map_item = ne.attrs_config(attr_maps)
                        attr_map_list.append(attr_map_item)
                    else:
                        attr_map = ne.attrs_config({})
                        attr_map_list.append(attr_map)
                    op_type = net_info['model']['operator'][node]['type']
                    op_config = ne.op_config(str(node), str(op_type), tensor_input[-1],
                                            tensor_output[-1], attr_map_list[-1])
                    op_configs.append(op_config)

                model_config = ne.model_config(net_info['model']['name'], op_configs)
                for node in net_info['model']['operator']['output_data']['input']:
                    output_list.append(node)
            else:
                model_config = net_info
                with open(model_config, 'r', encoding='utf-8') as f:
                    cfg = f.read()
                d = yaml.load(cfg, Loader=yaml.FullLoader)
                for node in d['model']['operator']:
                    if d['model']['operator'][node]['type'] == 'Output':
                        for tensor_name in d['model']['operator'][node]['input']:
                            output_list.append(tensor_name)
                    else:
                        continue
            if self._execution_options:
                model = ne.Model(model_config, weight_data, self._execution_options)
            else:
                model = ne.Model(model_config, weight_data)
            self._engine = [model, output_list, op_configs, tensor_output, tensor_input,
                            attr_map_list]

        if self._do_activation_mem_compression and self._max_input_shapes_list and \
           len(self._max_input_shapes_list) > 0:
            self._activation_mem_compression(self._max_input_shapes_list)
            self._refresh_max_input_shapes_list = False

    def inference(self, input_data):
        """The inference API of the neural engine."""
        # executor model init
        if self._refresh_execution_options or self._engine is None:
            self.engine_init()
            self._refresh_execution_options = False
        # update activation memory compresion when give different model input shapes
        if self._refresh_max_input_shapes_list and self._do_activation_mem_compression and self._engine:
            self.engine_init(refresh_model=False)
        output = self._engine[0].forward(input_data)
        index = 0
        output_dict = OrderedDict()
        for node in self._engine[1]:
            output_dict[node] = output[index]
            index += 1

        return output_dict

    def graph_init(self, config, weight_data=None, load_weight=False):
        """The initialization of the neural engine graph.

        Example::

            from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
            newgraph = Graph()
            newgraph.graph_init('./ir/conf.yaml', './ir/model.bin')
            out = newgraph.inference([input_0, input_1, input_2])
        """
        from ..ops import Tensor
        import yaml
        from .. import graph_utils as util
        import copy
        self._nodes = []
        self._node_id = {}
        self._engine = None
        yamlPath = os.path.join(config)
        cfg = None
        with open(yamlPath, 'r', encoding='utf-8') as f:
            cfg = f.read()
        d = yaml.load(cfg, Loader=yaml.FullLoader)
        bin_file = None
        if load_weight and weight_data != None:
            bin_file = open(weight_data, 'rb')
        output_list = []
        tensor_name_2_class = OrderedDict()
        for node in d['model']['operator']:
            op = None
            optype = d['model']['operator'][node]['type']
            if optype == 'Input':
                output_tensors = []
                for (tensor_name, attrs) in d['model']['operator'][node]['output'].items():
                    tensor_strides = None
                    if "strides" in attrs.keys():
                        tensor_strides = attrs["strides"]
                    tensor_shape = None
                    if "shape" in attrs.keys():
                        tensor_shape = attrs["shape"]
                    tensor_dtype = None
                    if "dtype" in attrs.keys():
                        tensor_dtype = attrs["dtype"]
                    tensor_location = None
                    tensor_data = None
                    if load_weight and 'location' in attrs.keys():
                        tensor_location = attrs['location']
                        bin_file.seek(tensor_location[0], 0)
                        tensor_data = copy.deepcopy(bin_file.read(tensor_location[1]))
                        DTYPES_DICT = {
                            "fp32": np.float32,
                            "s8": np.int8,
                            "s32": np.int32,
                            "u8": np.uint8,
                            "bf16": np.uint16,
                            'fp64': np.float64,
                        }
                        tensor_data = np.frombuffer(tensor_data, dtype=DTYPES_DICT[tensor_dtype]).\
                            reshape(tensor_shape)
                    tensorclass = Tensor()
                    if tensor_location == None:
                        tensorclass = Tensor(tensor_name, ['input_data'], [], tensor_shape,
                                             tensor_data, tensor_dtype, tensor_location)
                    else:
                        tensorclass = Tensor(tensor_name, [], [], tensor_shape, tensor_data,
                                             tensor_dtype, tensor_location)
                    tensor_name_2_class[tensor_name] = copy.deepcopy(tensorclass)
                    tensorclass.data = None
                    output_tensors.append(tensorclass)
                op = util.construct_node(node, 'Input', [], copy.deepcopy(output_tensors))

            elif optype == 'Output':
                input_tensors = []
                for tensor_name in d['model']['operator'][node]['input']:
                    if not load_weight:
                        output_list.append(tensor_name)
                    tensor = tensor_name_2_class[tensor_name]
                    input_tensors.append(tensor)
                op = util.construct_node(node, 'Output', copy.deepcopy(input_tensors))

            else:
                input_tensors = []
                output_tensors = []
                attr = {}
                for tensor_name in d['model']['operator'][node]['output']:
                    tensor = Tensor(tensor_name, [node])
                    tensor_name_2_class[tensor_name] = tensor
                    tensor.source_op.append(node)
                    output_tensors.append(tensor)

                A = d['model']['operator'][node]['input']
                for tensor_name in d['model']['operator'][node]['input']:
                    tensor = tensor_name_2_class[tensor_name]
                    tensor.dest_op.append(node)
                    input_tensors.append(tensor)
                if 'attr' in d['model']['operator'][node].keys():
                    attr = d['model']['operator'][node]['attr']

                op = util.construct_node(node, optype, copy.deepcopy(input_tensors),
                                         copy.deepcopy(output_tensors), attr)
            self.insert_nodes(len(self.nodes), [op])
        if not load_weight and weight_data:
            self._config_root = config
            self._weight_root = weight_data
        if bin_file:
            bin_file.close()

    def save(self, output_dir=None):
        """Save the model IR to the bin and ymal file."""
        logger.info("Start to emit the intermediate representation of model...")
        if output_dir is None:
            dir_name = os.getcwd()
            output_dir = os.path.join(dir_name, 'ir/')

        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # set the bin and yaml file name
        bin_file = os.path.join(output_dir, 'model.bin')
        yaml_file = os.path.join(output_dir, 'conf.yaml')

        # serialize_weight
        weight_data = self.weight_data
        with open(bin_file, 'wb') as f:
            f.write(weight_data)

        # serialize_network
        net_info = self.net_config
        with open(yaml_file, "w", encoding="utf-8") as f:
            # for write list, no use '-' to split the list, which is the default action in yaml
            def list_representer(dumper, data):
                return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

            # for write OrderedDict

            def dict_representer(dumper, data):
                return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())

            yaml.add_representer(list, list_representer)
            yaml.add_representer(OrderedDict, dict_representer)
            yaml.Dumper.ignore_aliases = lambda *args: True
            yaml.dump(net_info, f, default_flow_style=False, sort_keys=False)

        logger.info("Emit done...")

    def graph_dispatch(self, tune=True, inputs_shape=[]):
        """The graph dispatch for difference nodes."""
        sparse_nodes_name = self.get_sparse_nodes_name()
        if tune:
            logger.info("Tuning graph start ...")
            self._tune_onednn_graph(inputs_shape)
            self._tune_sparse_graph(inputs_shape, sparse_nodes_name)
            logger.info("Tuning graph end ...")
        else:
            # if not tune, map to sparse graph directly
            self.transpose_mode_int8(sparse_nodes_name)

    def _tune_onednn_graph(self, inputs_shape=[]):
        """The onednn graph tuning."""
        onednn_graph_nodes_map = self._get_onednn_graph_nodes()
        if onednn_graph_nodes_map == {"InnerProduct": [], "Softmax": []}:
            pass
        else:
            onednn_graph_nodes_name_list = self._generate_onednn_graph_nodes_name_list(
                onednn_graph_nodes_map)
            golden_onednn_graph_nodes_name = []
            min_latency = float("inf")
            for onednn_graph_nodes_name in onednn_graph_nodes_name_list:
                curr_latency = float("inf")
                try:
                    curr_model = copy.deepcopy(self)
                    curr_model._generate_onednn_graph_nodes(onednn_graph_nodes_name)
                    curr_result, curr_latency = curr_model._get_latency(inputs_shape)
                except:
                    logger.warning("Graph can not be inferenced, please check the graph!")
                # update min latency and transpose nodes name
                if curr_latency < min_latency:
                    min_latency = curr_latency
                    golden_onednn_graph_nodes_name = onednn_graph_nodes_name
            self._generate_onednn_graph_nodes(golden_onednn_graph_nodes_name)

    def _get_onednn_graph_nodes(self):
        """Onednn graph only support fp32 inner_product and softmax."""
        onednn_graph_nodes_map = {"InnerProduct": [], "Softmax": []}
        for node in self.nodes:
            if node.op_type == "InnerProduct":
                weight = node.input_tensors[1]
                if type(weight.data) == np.ndarray and \
                        weight.data.dtype == "float32":
                    onednn_graph_nodes_map["InnerProduct"].append(node.name)
            elif node.op_type == "Softmax":
                if node.attr.get("output_dtype", "float32") == "float32":
                    onednn_graph_nodes_map["Softmax"].append(node.name)
        return onednn_graph_nodes_map

    def _generate_onednn_graph_nodes_name_list(self, onednn_graph_nodes_map):
        """Strategy.

        1.softmax: all nodes map to onednn graph or not
        2.innerproduct: tune accorording weight shape
        """
        ip_nodes_name_list = self._generate_transpose_nodes_name_list(
            onednn_graph_nodes_map["InnerProduct"])
        onednn_graph_nodes_name_list = []
        for ip_nodes_name in ip_nodes_name_list:
            onednn_graph_nodes_name_list.append(ip_nodes_name)
            onednn_graph_nodes_name_list.append(ip_nodes_name + onednn_graph_nodes_map["Softmax"])
        return onednn_graph_nodes_name_list

    def _generate_onednn_graph_nodes(self, onednn_graph_nodes_name):
        """The onednn graph ndoes generation."""
        for node in self.nodes:
            if node.name in onednn_graph_nodes_name:
                if node.op_type == "InnerProduct":
                    node.op_type = "InnerProductGraph"
                elif node.op_type == "Softmax":
                    node.op_type = "SoftmaxGraph"

    def _tune_sparse_graph(self, inputs_shape=[], sparse_nodes_name=[]):
        """The sparse graph tuning."""
        if sparse_nodes_name == []:
            pass
        else:
            trans_nodes_name_list = self._generate_transpose_nodes_name_list(sparse_nodes_name)
            golden_trans_nodes_name = []
            min_latency = float("inf")
            for trans_nodes_name in trans_nodes_name_list:
                curr_latency = float("inf")
                try:
                    curr_model = copy.deepcopy(self)
                    curr_model.transpose_mode_int8(trans_nodes_name)
                    curr_result, curr_latency = curr_model._get_latency(inputs_shape)
                except:
                    logger.warning("Graph can not be inferenced, please check the graph!")
                # update min latency and transpose nodes name
                if curr_latency < min_latency:
                    min_latency = curr_latency
                    golden_trans_nodes_name = trans_nodes_name
            self.transpose_mode_int8(golden_trans_nodes_name)

    def get_sparse_nodes_name(self, threshold=0.7):
        """According to the sparsity threshold to list all sparse nodes."""

        def get_zero_ratio(matrix, block):
            sparse_ratio = -1
            if matrix.ndim == 2 and len(block) == 2:
                zero_block_count = 0
                block_row = int(matrix.shape[0] / block[0])
                block_col = int(matrix.shape[1] / block[1])
                for mr in range(block_row):
                    for mc in range(block_col):
                        is_zero_block = True
                        for br in range(block[0]):
                            for bc in range(block[1]):
                                if matrix[mr * block[0] + br][mc * block[1] + bc] != 0:
                                    is_zero_block = False
                                    break
                            if not is_zero_block:
                                break
                        if is_zero_block == True:
                            zero_block_count += 1
                zero_ratio = float(zero_block_count) / (block_row * block_col)
            return zero_ratio

        sparse_nodes_name = []
        for node in self.nodes:
            if node.op_type == "InnerProduct":
                # sparse kernel limitation:
                # 1. int8
                # 2. sparse_ratio > 0.7(1*4)
                # 3. output channel of weight_shape = 4x
                # 4. post op != tanh
                if 'append_op' not in node.attr \
                   or ('append_op' in node.attr and
                       node.attr['append_op'] != 'tanh'):
                    weight = node.input_tensors[1]
                    if type(weight.data) == np.ndarray and \
                        (weight.data.dtype == 'int8'
                         or weight.data.dtype == 'uint8') \
                            and weight.data.shape[1] % 4 == 0:  # 1*4 sparse block
                        zero_ratio = get_zero_ratio(weight.data, [1, 4])
                        if zero_ratio >= threshold:
                            sparse_nodes_name.append(node.name)

        return sparse_nodes_name

    def _generate_transpose_nodes_name_list(self, sparse_nodes_name):
        """The name list of transpose nodes generation."""
        transpose_nodes_list = []
        if sparse_nodes_name == []:
            return transpose_nodes_list
        # switch the nodes which has the same weight shape and pose op
        weight_shape_map = {}
        for node in self.nodes:
            if node.name in sparse_nodes_name:
                weight = node.input_tensors[1]
                # list to tuple for dict key
                weight_shape = tuple(weight.shape)
                if weight_shape in weight_shape_map.keys():
                    weight_shape_map[weight_shape].append(node.name)
                else:
                    weight_shape_map[weight_shape] = [node.name]

        def brgd(n):
            """Binary reflected gray code to generate the all combinations fo the n elements."""
            if n == 1:
                return ["0", "1"]
            L1 = brgd(n - 1)
            L2 = copy.deepcopy(L1)
            L2.reverse()
            L1 = ["0" + l for l in L1]
            L2 = ["1" + l for l in L2]
            L = L1 + L2
            return L

        transpose_mask_list = brgd(len(weight_shape_map))
        for transpose_mask in transpose_mask_list:
            transpose_nodes = []
            for idx, weight_shape in enumerate(weight_shape_map):
                if transpose_mask[idx] == "1":
                    transpose_nodes += weight_shape_map[weight_shape]
            transpose_nodes_list.append(transpose_nodes)

        return transpose_nodes_list

    def _generate_inputs(self, inputs_shape=[]):
        """The input of nodes generation."""
        dtype_map = {
            "float32": np.float32,
            "int8": np.int8,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
        }
        inputs = []
        id = 0
        for node in self.nodes:
            if node.op_type == "Input":
                for tensor in node.output_tensors:
                    if not isinstance(tensor.data, np.ndarray):
                        if inputs_shape == []:
                            shape = [16 for s in tensor.shape if s == -1]
                        else:
                            shape = inputs_shape[id]
                        dtype = dtype_map[tensor.dtype]
                        input = np.random.uniform(low=0, high=10, size=shape).astype(dtype)
                        inputs.append(input)
                        id += 1
        return inputs

    def _get_latency(self, inputs_shape=[], iterations=10, warm_up=5):
        """Calculate the latency."""
        inputs = self._generate_inputs(inputs_shape)
        iter_latency = []
        for _ in range(iterations):
            start_time = time.time()
            result = self.inference(inputs)
            end_time = time.time()
            iter_latency.append(end_time - start_time)
        latency = np.array(iter_latency[warm_up:]).mean()
        return result, latency

    def transpose_mode_int8(self, node_name_list=None):
        """Transpose innerproduct nodes to the sparse mode by inserting reorder nodes."""
        from ..ops import Tensor
        from .. import graph_utils as util
        import copy
        logger.info("Start to transpose_mode_int8 ......")
        reorder_dict = {}

        def _modify_attr_perm(node):
            """Transpose the node attr perm."""
            if node.attr.get("src0_perm") == '0,2,1,3' and node.attr.get("src1_perm") == '0,2,3,1':
                node.attr["src0_perm"] = '2,0,3,1'
                node.attr["src1_perm"] = '2,0,1,3'
            if node.attr.get("src1_perm") == '0,2,1,3' and node.attr.get("dst_perm") == '0,2,1,3':
                node.attr["src1_perm"] = '2,0,3,1'
                node.attr["dst_perm"] = '1,3,0,2'
            if 'reshape' in node.attr:
                _reorder_shape_list(node, 'reshape')

        def _innerproduct_type_check(node):
            """Check the innerproduct node type."""
            innerproduct_type = {
                # general_node
                "general": "general",

                # InnerProduct Nodes
                "QKV_innerproduct": 'add_innerproduct_0',
                'output_dense_bias': 'add_innerproduct_1',
                'intermediate_dense_mul': 'mul_innerproduct_0',

                # Matmul Nodes
                'matmul_type': 'matmul_0',
            }

            if node.op_type == "InnerProduct":
                if 'append_op' in node.attr and node.attr['append_op'] == 'sum':
                    return innerproduct_type['output_dense_bias']
                if 'append_op' in node.attr and node.attr['append_op'] == 'gelu_tanh':
                    return innerproduct_type['intermediate_dense_mul']
                else:
                    return innerproduct_type['QKV_innerproduct']

            if node.op_type == 'Matmul':
                return innerproduct_type['matmul_type']
            else:
                return innerproduct_type['general']

        def _create_new_attr(node, mode=None):
            """The output_dtype of the inserted reorder node is the same as the input node dtype if dtype exists."""
            if mode == 'Reorder_Post':
                new_attr = OrderedDict({'src_perm': '0,1', 'dst_perm': '1,0'})
                if 'output_dtype' in node.attr:
                    new_attr['output_dtype'] = node.attr['output_dtype']
                return new_attr

            if mode == 'Reorder_Recover':
                if 'reshape' in node.attr and 'reshape_dims' in node.attr:
                    value_list = node.attr['reshape'].split(',')
                    if len(value_list) == 4:
                        new_attr = OrderedDict({'src_perm': '0,1,2,3', 'dst_perm': '2,3,0,1'})
                    else:
                        logger.info('The length of value_list is wrong')
                else:
                    new_attr = OrderedDict({'src_perm': '0,1', 'dst_perm': '1,0'})

                if 'output_dtype' in node.attr:
                    new_attr['output_dtype'] = node.attr['output_dtype']

                return new_attr

            return False

        def _dfs_search(node, target_node_type):
            """Use the deep-first algorithm to search the node accroding to the its type."""
            target_node = []

            def dfs(node):
                if node.op_type == target_node_type:
                    return node
                if node.output_tensors[0].dest_op == []:
                    return None

                tmp_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])
                if tmp_node.op_type != target_node_type:
                    if dfs(tmp_node) == None:
                        return None
                else:
                    target_node.append(tmp_node)
                    return tmp_node

            dfs(node)
            if target_node == []:
                return None
            else:
                return target_node[0]

        def _reorder_shape_list(node, attr_name='dst_shape'):
            """Transpose the shape attribute."""
            value_list = node.attr[attr_name].split(',')
            if len(value_list) == 4:
                value = value_list[2] + ',' + value_list[3] + \
                    ',' + value_list[0] + ', ' + value_list[1]

            if len(value_list) == 2:
                value = value_list[1] + ',' + value_list[0]

            node.attr[attr_name] = value

        def _reorder_node_insert(node, idx, insert_pos=None):
            """Reorder node insert.

            Args:
                node: the current innerproduct node.
                idx: the position of variables that need to be transposed in node.input_tensors.
                insert_pos: the position of insert.
            """
            # check the node whether has been reordered
            for index_node, reorder_node in reorder_dict.items():
                if node.input_tensors[idx].name == reorder_node.input_tensors[0].name:
                    # the variable that need to be transposed has been reordered.
                    node.input_tensors[idx] = reorder_node.output_tensors[0]
                    # append the current node to the reorder node output_tensors[0].dest_op
                    reorder_node.output_tensors[0].dest_op.append(node.name)
                    # no need to insert reorder node again
                    return

            pre_node = self.get_node_by_name(node.input_tensors[idx].source_op[0])
            input_name = pre_node.output_tensors[0].name
            reorder_node_name = node.name + "_Reorder_Post_" + str(idx)
            reorder_output_tensor = Tensor(name=input_name + "_reorder",
                                           source_op=[reorder_node_name],
                                           dest_op=[node.name],
                                           dtype=node.output_tensors[0].dtype)
            node.input_tensors[idx] = reorder_output_tensor

            new_attr = _create_new_attr(pre_node, 'Reorder_Post')
            reorder_node = util.construct_node(node_name=reorder_node_name,
                                               op_type='Reorder',
                                               input_tensors=[pre_node.output_tensors[0]],
                                               output_tensors=[reorder_output_tensor],
                                               attr=new_attr)

            if insert_pos != None:
                reorder_dict[insert_pos.name] = reorder_node
            else:
                reorder_dict[node.name] = reorder_node
            return reorder_node

        def _swap_innertproduct_input(node, data_swap_list=[0, 1], swap_list_2=[], swap_list_3=[]):
            """Modify innertproduct nodes and the folloing reshape nodes."""
            input_0 = data_swap_list[0]
            input_1 = data_swap_list[1]
            # swap(input_0, input1)
            node.input_tensors[input_1].data = np.ascontiguousarray(
                node.input_tensors[input_1].data.T)
            node.input_tensors[input_1].shape = list(node.input_tensors[input_1].data.shape)
            tmp_node = copy.deepcopy(node.input_tensors[input_1])
            node.input_tensors[input_1] = node.input_tensors[input_0]
            node.input_tensors[input_0] = tmp_node

            input_3 = swap_list_2[0]
            input_5 = swap_list_2[1]
            tmp_node = copy.deepcopy(node.input_tensors[input_3])
            node.input_tensors[input_3] = node.input_tensors[input_5]
            node.input_tensors[input_5] = tmp_node

            input_4 = swap_list_3[0]
            input_6 = swap_list_3[1]
            tmp_node = copy.deepcopy(node.input_tensors[input_4])
            node.input_tensors[input_4] = node.input_tensors[input_6]
            node.input_tensors[input_6] = tmp_node
            if 'src1_perm' in node.attr:
                node.attr.pop('src1_perm')
            if 'reshape' in node.attr:
                _reorder_shape_list(node, 'reshape')

        def _modify_post_node_attr(post_node, attr_name='axis'):
            """Modify the post node attribute."""
            if attr_name in post_node.attr:
                if post_node.attr[attr_name] == 0:
                    post_node.attr[attr_name] = 1
                elif post_node.attr[attr_name] == 1:
                    post_node.attr[attr_name] = 0

        def _reorder_recover_node_insert(node, pre_node=None):
            """Reorder_recover node insert.

            Args:
                node: the current innerproduct node.
                pre_node: the first predecessor node of the reorder_recover node.
            """
            post_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])
            if len(node.output_tensors[0].dest_op) != 1:
                logger.info("the post node is not only one.")
            reorder_recover_node_name = node.name + '_Reorder_Recover'

            if pre_node != None:
                pre_node_output = pre_node.output_tensors[0]
            else:
                pre_node_output = node.output_tensors[0]

            reorder_output_tensor = Tensor(name=pre_node_output.name + "_recover",
                                           source_op=[reorder_recover_node_name],
                                           dest_op=[post_node.name],
                                           dtype=node.output_tensors[0].dtype)

            new_attr = _create_new_attr(node, 'Reorder_Recover')
            reorder_recover_node = util.construct_node(node_name=reorder_recover_node_name,
                                                       op_type='Reorder',
                                                       input_tensors=[pre_node_output],
                                                       output_tensors=[reorder_output_tensor],
                                                       attr=new_attr)

            insert_idx = self.get_node_id(post_node.name)
            self.insert_nodes(insert_idx, [reorder_recover_node])
            if len(node.output_tensors[0].dest_op) != 1:
                logger.info("the post node is not only one.")

            idx = 0
            for _ in post_node.input_tensors:
                if node.output_tensors[0].name == _.name:
                    break
                else:
                    idx = idx + 1

            post_node.input_tensors[idx] = reorder_output_tensor
            _modify_post_node_attr(post_node, 'axis')
            return reorder_recover_node

        def _modify_post_node_input_tensor(post_node, node, node_input_tensors_idx=0):
            """Modify the input tensors of the post node.

            Args:
                post_node: the post node of the second parameter
                node: the source of input_tensors
                node_input_tensors_idx: assign the index of node.input_tensors to post_node.input_tensors
            """
            idx = 0
            for _ in post_node.input_tensors:
                if node.output_tensors[0].name == _.name:
                    break
                else:
                    idx = idx + 1
            post_node.input_tensors[idx] = node.input_tensors[node_input_tensors_idx]
            return

        def _del_current_node_and_modify_post_node(node):
            """Delete the current node and then modify input tensors of the post node."""
            pre_node = self.get_node_by_name(node.input_tensors[0].source_op[0])
            for post_node_name in node.output_tensors[0].dest_op:
                post_node = self.get_node_by_name(post_node_name)
                _modify_post_node_input_tensor(post_node, node, 0)
            self.remove_nodes([node.name])
            for post_node_name in node.output_tensors[0].dest_op:
                post_node = self.get_node_by_name(post_node_name)
                _modify_post_node_attr(post_node, 'axis')
                pre_node.output_tensors[0].dest_op.append(post_node.name)

        def _node_name_list_convert(node_name_list):
            """List all innerproduct node types."""
            merge_matmul_node_name_list = []
            innerproduct_type = []
            for node_name in node_name_list:
                node = self.get_node_by_name(node_name)
                node_type = _innerproduct_type_check(node)
                innerproduct_type.append(node_type)
                if node_type == 'add_innerproduct_0' or node_type == 'mul_innerproduct_0':
                    if 'merge_matmul' in node.name:
                        merge_matmul_node_name_list.append(node_name)

            return innerproduct_type, merge_matmul_node_name_list

        def _pattern_matching(innerproduct_type_list, pattern):
            """Matching the pattern accroding to innerproduct node types."""
            matched_idx = []
            pattern_length = len(pattern)

            left = 0
            N = len(innerproduct_type_list)
            while left < N:

                def _check(left):
                    if pattern == innerproduct_type_list[left:left + pattern_length]:
                        return True
                    return False

                if _check(left) == True:
                    matched_idx.append(left)
                left += 1

            return matched_idx

        def _matched_node_name_list(node_name_list, matched_idx, range=4):
            """The list of matched nodes."""
            tmp_node_name_list = []
            for node_idx in matched_idx:
                tmp_node_name_list.append(node_name_list[node_idx:node_idx + range])
            node_name_list = [i for item in tmp_node_name_list for i in item]
            return node_name_list

        def _search_layernorm_fusion_node(ffn_lin_node_name_list,
                                          start_idx,
                                          range,
                                          node_name_list=[]):
            """Search nodes related to layernorm that can be fused."""
            layernorm_node = []
            # pcik all add_innerproduct_1 nodes from ffn_lin_node_name_list
            for node_name in ffn_lin_node_name_list[start_idx:][::range]:
                # The node_name_list here is also to identify the merge matmul pattern.
                if node_name_list != []:
                    split_node_name = self.get_node_by_name(node_name).output_tensors[0].dest_op[0]
                    length_of_split_node_output = len(
                        self.get_node_by_name(split_node_name).output_tensors)
                    if length_of_split_node_output == 3:
                        node_name = node_name_list[node_name_list.index(node_name) + 1]
                    elif length_of_split_node_output == 2:
                        node_name = node_name_list[node_name_list.index(node_name) + 2]
                    else:
                        logger.warning('The output of the split node is not expected.')

                layernorm_node_name = self.get_node_by_name(
                    node_name).input_tensors[3].source_op[0]
                node = self.get_node_by_name(layernorm_node_name)
                if node.op_type != 'LayerNorm':
                    logger.warning('Post Process Pattern matching is not expected')
                else:
                    layernorm_node.append(node.name)
            return layernorm_node

        def _consecutive_reorder_fusion():
            """Fusion 1.

            eliminate the two reorder nodes if a tensor passes through reorder_recover + reorder_post consecutively
            """
            for node in self._nodes:
                if node.op_type == 'Reorder' and 'recover_reorder' in node.output_tensors[0].name:
                    if 'Reorder_Post' in node.name:
                        pre_node = self.get_node_by_name(node.input_tensors[0].source_op[0])
                        if 'Reorder_Recover' in pre_node.name and pre_node.op_type == 'Reorder':
                            post_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])
                            idx = 0
                            for _ in post_node.input_tensors:
                                if node.output_tensors[0].name == _.name:
                                    break
                                else:
                                    idx = idx + 1
                            post_node.input_tensors[idx] = pre_node.input_tensors[0]
                            self.remove_nodes([pre_node.name, node.name])
                            pre_node.input_tensors[0].dest_op.append(post_node.name)

        def _reorder_post_fusion():
            """Fusion 2.

            This fusion is used to place reorder_post nodes before the quantize node.
            Conditions:
                Only fusion if all QKV nodes meet the sparsity ratio

            E.g.
                original: quantize + reorder_post + innerproduct: K + innerproduct: Q + innerproduct: v
                fusion: reorder_post + quantize + innerproduct: K + innerproduct: Q + innerproduct: v
            """

            # check the current reorder_post_node whether need place reorder_post nodes before the quantize node
            def _check_QKV_fusion(reorder_post_node):
                innerproduct_node = self.get_node_by_name(
                    reorder_post_node.output_tensors[0].dest_op[0])
                node_type = _innerproduct_type_check(innerproduct_node)

                # This branch is used to check type of the attention out node.
                if node_type == 'mul_innerproduct_0':
                    return True

                # The dest_op of reorder_post_node.output_tensors[0] are QKV nodes.
                for QKV_node_name in reorder_post_node.output_tensors[0].dest_op:
                    if QKV_node_name in QKV_node_name_list:
                        continue
                    else:
                        return False
                # only when all QKV nodes are in the QKV node name list return true
                return True

            def _check_merged_matmul(reorder_post_node):
                """Check if the node is in the merged matmul node name list."""
                innerproduct_node = self.get_node_by_name(
                    reorder_post_node.output_tensors[0].dest_op[0])
                if innerproduct_node.name not in merge_matmul_node_name_list:
                    return False
                return True

            for node in self._nodes:
                # search all Reorder_Post nodes
                if node.op_type == 'Reorder' and 'Reorder_Post' in node.name:
                    if _check_QKV_fusion(node) == False:
                        if _check_merged_matmul(node) == False:
                            continue

                    pre_node = self.get_node_by_name(node.input_tensors[0].source_op[0])
                    if pre_node.op_type == 'Quantize':
                        # swap the pre_node and the current node
                        for post_node_name in node.output_tensors[0].dest_op:
                            post_node = self.get_node_by_name(post_node_name)
                            _modify_post_node_input_tensor(post_node, node, 0)
                        self.remove_nodes([node.name])
                        reorder_node = _reorder_node_insert(pre_node, 0)
                        layernorm_node = self.get_node_by_name(
                            reorder_node.input_tensors[0].source_op[0])
                        if layernorm_node.op_type != 'LayerNorm':
                            logger.warning('The node.op_type = {} is not expected.'.format(
                                layernorm_node.op_type))

                        # the dest_op of layernorm_node.output_tensors[0] is reorder_post node for attention out node.
                        if 'Reorder_Post' in layernorm_node.output_tensors[0].dest_op[0]:
                            reorder_post_node = self.get_node_by_name(
                                layernorm_node.output_tensors[0].dest_op[0])
                            attention_out_node = self.get_node_by_name(
                                reorder_post_node.output_tensors[0].dest_op[0])
                            self.remove_nodes([reorder_post_node.name])
                            # append the new reorder_node
                            layernorm_node.output_tensors[0].dest_op.append(reorder_node.name)
                            reorder_node.output_tensors[0].dest_op.append(attention_out_node.name)

            for node_name in reorder_dict:
                insert_idx = self.get_node_id(node_name)
                self.insert_nodes(insert_idx, [reorder_dict[node_name]])

        def _reorder_recover_fusion():
            """Fusion 3: place the reorder_recover nodes after reshape and matmul nodes.

            This fusion deletes the reorder_recover nodes that follow Q, K, V and attention out nodes
            and the post modify add_matmul and transpose_matmul nodes.
            """
            for node in self._nodes:
                # step1: delte all recover nodes of innerProduct nodes and modify reshape inputs
                node_type = _innerproduct_type_check(node)
                if node_type == 'add_innerproduct_0' and node.name in QKV_node_name_list:
                    post_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])
                    # innerproduct + reorder_recover + reshape
                    if 'Reorder_Recover' in post_node.name:
                        _del_current_node_and_modify_post_node(post_node)

                        # This conditional branch will be skipped if the innerproduct node fuses the reshape node.
                        reshape_node = self.get_node_by_name(
                            post_node.output_tensors[0].dest_op[0])
                        if reshape_node.op_type == 'Reshape':
                            _reorder_shape_list(reshape_node)

                        # step2 : modify add_matmul and transpose_matmul nodes
                        target_node = _dfs_search(node, 'Matmul')
                        if _innerproduct_type_check(target_node) == 'matmul_0':

                            def _transpose_and_matmul_nodes_modification(node):
                                _modify_attr_perm(node)
                                reshape_node = self.get_node_by_name(
                                    node.output_tensors[0].dest_op[0])
                                if reshape_node.op_type == "Reshape":
                                    _reorder_shape_list(reshape_node)

                                reorder_node = _dfs_search(node, 'Reorder')
                                innerproduct_node = self.get_node_by_name(
                                    reorder_node.output_tensors[0].dest_op[0])
                                if 'Reorder_Post' in reorder_node.name and innerproduct_node.op_type == "InnerProduct":
                                    _del_current_node_and_modify_post_node(reorder_node)

                            _transpose_and_matmul_nodes_modification(target_node)
                    else:
                        logger.warning('The node op_type == {} is not expected.'.format(
                            post_node.name))

        def _layernorm_reorder_fusion():
            """The fusion of layernorm reorder."""
            for node in self._nodes:
                if node.op_type == 'LayerNorm' and node.name in layernorm_fusion_node:
                    reorder_recover_node = self.get_node_by_name(
                        node.input_tensors[0].source_op[0])
                    reorder_post_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])

                    if 'Reorder_Recover' in reorder_recover_node.name and 'Reorder_Post' in reorder_post_node.name:
                        node.attr['transpose_mode'] = '1, 0'
                        _del_current_node_and_modify_post_node(reorder_recover_node)
                        _del_current_node_and_modify_post_node(reorder_post_node)

        def _merged_matmul_fusion():
            """The fusition of merged matmul."""

            def _transpose_matmul_modification(node):
                reorder_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])
                attention_out_node = self.get_node_by_name(
                    reorder_node.output_tensors[0].dest_op[0])
                if attention_out_node.op_type == "InnerProduct":
                    _del_current_node_and_modify_post_node(reorder_node)
                else:
                    logger.warning('The attention_out_node.op_type = {} is not expected'.format(
                        attention_out_node.op_type))

            for node_name in node_name_list:
                node = self.get_node_by_name(node_name)

                if 'merge_matmul' not in node.name:
                    continue

                reorder_recover_node = self.get_node_by_name(node.output_tensors[0].dest_op[0])

                if 'Reorder_Recover' in reorder_recover_node.name:
                    _del_current_node_and_modify_post_node(reorder_recover_node)
                    split_node = self.get_node_by_name(
                        reorder_recover_node.output_tensors[0].dest_op[0])

                    def mergedQK_modify(split_node):
                        for post_tensor in split_node.output_tensors:
                            post_node = self.get_node_by_name(post_tensor.dest_op[0])

                            # Reshape + Matmul + Softmax
                            if post_node.op_type == 'Reshape':
                                _reorder_shape_list(post_node)
                                post_node_of_reshape = self.get_node_by_name(
                                    post_node.output_tensors[0].dest_op[0])
                                if post_node_of_reshape.op_type == 'Matmul':
                                    _modify_attr_perm(post_node_of_reshape)

                        # InnerProduct V + Matmul + InnerProduct Attention_output
                        # delete the recover node of InnerProduct v
                        split_node_id = self.get_node_id(split_node.name)
                        next_node_id = split_node_id + 1
                        v = self._nodes[next_node_id]

                        if v.op_type == 'InnerProduct':
                            recover_node = self.get_node_by_name(v.output_tensors[0].dest_op[0])
                            _del_current_node_and_modify_post_node(recover_node)
                        else:
                            logger.warning('The v.op_type is not expected.')

                        # modify the matmul node
                        post_node_of_v = self.get_node_by_name(v.output_tensors[0].dest_op[0])
                        # This branch is only used if there is no reshaping + innerproduct fusion.
                        if post_node_of_v.op_type == 'Reshape':
                            _reorder_shape_list(post_node_of_v)
                            post_node_of_reshape = self.get_node_by_name(
                                post_node_of_v.output_tensors[0].dest_op[0])
                            if post_node_of_reshape.op_type == 'Matmul':
                                _modify_attr_perm(post_node_of_reshape)
                                tmp_node = self.get_node_by_name(
                                    post_node_of_reshape.output_tensors[0].dest_op[0])
                                if tmp_node.op_type == "Reshape":
                                    _reorder_shape_list(tmp_node)
                                    _transpose_matmul_modification(tmp_node)
                        elif post_node_of_v.op_type == 'Matmul':
                            _modify_attr_perm(post_node_of_v)
                            _transpose_matmul_modification(post_node_of_v)
                        else:
                            logger.warning('The mergedQK fusion is not expected.')

                    def mergedQKV_modify(split_node):
                        for post_tensor in split_node.output_tensors:
                            post_node = self.get_node_by_name(post_tensor.dest_op[0])
                            # 3 * Reshape + Matmul + Innerproduct_Sum
                            if post_node.op_type == 'Reshape':
                                _reorder_shape_list(post_node)
                                matmul_node = self.get_node_by_name(
                                    post_node.output_tensors[0].dest_op[0])
                                if matmul_node.op_type == 'Matmul':
                                    _modify_attr_perm(matmul_node)
                                    post_node_of_matmul = self.get_node_by_name(
                                        matmul_node.output_tensors[0].dest_op[0])
                                    # post_node_of_matmul could also be softmax.
                                    if post_node_of_matmul.op_type == 'Reorder':
                                        _del_current_node_and_modify_post_node(post_node_of_matmul)
                                else:
                                    logger.warning('The post_node_of_reshape is not Matmul.')

                    if len(split_node.output_tensors) == 2:
                        mergedQK_modify(split_node)

                    if len(split_node.output_tensors) == 3:
                        mergedQKV_modify(split_node)

        for node in self._nodes:
            if node.op_type == 'Reorder':
                logger.info('The current IR is reordered, No need to do transpose.')
                return

        if node_name_list == None:
            logger.info("The node_name_list is None. Start to get sparse nodes name...")
            node_name_list = self.get_sparse_nodes_name()

        logger.info('node_name_list: %s', node_name_list)

        if node_name_list == []:
            logger.info("The node_name_list is [].")
            return

        innerproduct_type_list, merge_matmul_node_name_list = _node_name_list_convert(
            node_name_list)
        logger.debug('merge_matmul_node_name_list = {}'.format(merge_matmul_node_name_list))

        QKV_and_AttentionOut = [
            'add_innerproduct_0', 'add_innerproduct_0', 'add_innerproduct_0', 'add_innerproduct_1'
        ]
        matched_idx = _pattern_matching(innerproduct_type_list, QKV_and_AttentionOut)
        QKV_node_name_list = _matched_node_name_list(node_name_list, matched_idx, 4)
        logger.debug('QKV_node_name_list = {}'.format(QKV_node_name_list))

        ffn_lin = ['mul_innerproduct_0', 'add_innerproduct_1']
        matched_idx = _pattern_matching(innerproduct_type_list, ffn_lin)
        ffn_lin_node_name_list = _matched_node_name_list(node_name_list, matched_idx, 2)
        logger.debug('ffn_lin_node_name_list = {}'.format(ffn_lin_node_name_list))

        layernorm_fusion_node = _search_layernorm_fusion_node(ffn_lin_node_name_list, 1, 2)
        layernorm_fusion_node += _search_layernorm_fusion_node(QKV_node_name_list, 3, 4)
        if merge_matmul_node_name_list != []:
            layernorm_fusion_node += _search_layernorm_fusion_node(merge_matmul_node_name_list, 0,
                                                                   1, node_name_list)
        logger.debug('layernorm_fusion_node = {}'.format(layernorm_fusion_node))

        for node_name in node_name_list:
            node = self.get_node_by_name(node_name)
            node_type = _innerproduct_type_check(node)

            if node_type == 'general':
                continue

            if node_type == 'add_innerproduct_0' or node_type == 'mul_innerproduct_0':
                _reorder_node_insert(node, 0)
                _swap_innertproduct_input(node, [0, 1], [3, 5], [4, 6])
                _reorder_recover_node_insert(node)

            if node_type == 'add_innerproduct_1':
                reorder_node = _reorder_node_insert(node, 0)
                reorder_node.attr['output_dtype'] = 'u8'
                _reorder_node_insert(node, 3, reorder_node)

                _swap_innertproduct_input(node, [0, 1], [4, 6], [5, 7])
                _reorder_recover_node_insert(node)

        for node_name in reorder_dict:
            insert_idx = self.get_node_id(node_name)
            self.insert_nodes(insert_idx, [reorder_dict[node_name]])

        _consecutive_reorder_fusion()

        reorder_dict = {}
        _reorder_post_fusion()

        _reorder_recover_fusion()

        _layernorm_reorder_fusion()

        if merge_matmul_node_name_list == []:
            logger.info('No mergedMatmul nodes, No need do merged matmul fusion')
            return

        reorder_dict = {}
        _merged_matmul_fusion()

        logger.info("Transpose_mode_int8 done")

    def _activation_mem_compression(self, input_shapes_list):
        if self._engine and input_shapes_list and len(input_shapes_list) > 0:
            self._engine[0].activation_mem_compression(input_shapes_list)
        else:
            logger.warning("Skip activation_mem_compression due to empty engine model or " \
                               "empty input_shapes.")
