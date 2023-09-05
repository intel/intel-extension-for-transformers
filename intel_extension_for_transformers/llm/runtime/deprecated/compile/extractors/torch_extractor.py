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

"""The neural engine torch extractor file."""

from ..graph_utils import LazyImport
from .. import logger
from ..graph.graph import Graph
from ..ops.op import OPERATORS
from ..onnx_utils import graph_node_names_details
from ..graph_utils import names_from_input, get_data_dtype, construct_node
from ..ops.tensor import Tensor
from ..torch_utils import get_node_name, op_maps
from .. import graph_utils as util

torch = LazyImport('torch')

def removeUnusedNode(graph, unused_nodes):
    remove_list = []
    for node in graph.nodes():
        if node.kind() in unused_nodes:
            in_val = node.inputsAt(0)
            out_val = node.outputsAt(0)
            out_val.replaceAllUsesWith(in_val)
            remove_list.append(node)

    for node in remove_list:
        node.destroy()

def fuse_padding_seq(graph):
    old_g_0 = """
            graph(%input_ids.1, %attention_mask.1, %3, %4, %5, %6, %7, %8, %9, %10):
                %11 : int = aten::size(%input_ids.1, %9)
                %12 : int[] = prim::ListConstruct(%11, %8)
                %attention_mask0.1 : Tensor = aten::view(%attention_mask.1, %12)
                %14 : Tensor = aten::slice(%attention_mask0.1, %9, %9, %7, %6)
                %15 : Tensor = aten::unsqueeze(%14, %6)
                %16 : Tensor = aten::unsqueeze(%15, %5)
                %17 : Tensor = aten::slice(%16, %4, %9, %7, %6)
                %18 : Tensor = aten::rsub(%17, %3, %6)
                %19 : Tensor = aten::mul(%18, %10)
                return (%19)
            """
    new_g_0 = """
            graph(%input_ids.1, %attention_mask.1, %3, %4, %5, %6, %7, %8, %9, %10):
                %19 = prim::padding_sequence(%attention_mask.1, %10)
                return (%19)
            """
    old_g_1 = """
            graph(%attention_mask.1, %11, %12, %13, %14, %15, %17, %8):
                %231 : Tensor = aten::slice(%attention_mask.1, %11, %11, %12, %13)
                %232 : Tensor = aten::unsqueeze(%231, %13)
                %233 : Tensor = aten::unsqueeze(%232, %14)
                %extended_attention_mask.1 : Tensor = aten::slice(%233, %15, %11, %12, %13)
                %236 : Tensor = aten::rsub(%extended_attention_mask.1, %17, %13)
                %attention_mask0.1 : Tensor = aten::mul(%236, %8)
                return (%attention_mask0.1)
            """
    new_g_1 = """
            graph(%attention_mask.1, %11, %12, %13, %14, %15, %17, %8):
                %attention_mask0.1 = prim::padding_sequence(%attention_mask.1, %8)
                return (%attention_mask0.1)
            """
    graph_pairs = [(old_g_0, new_g_0), (old_g_1, new_g_1)]
    for g_pair in graph_pairs:
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(g_pair[0], g_pair[1], graph)

def fuse_position_ids(graph):
    old_g_0 = """
            graph(%input_ids.1, %13, %21, %7, %11):
                %238 : int = aten::size(%input_ids.1, %13)
                %240 : Tensor = aten::add(%238, %21, %13)
                %input.13 : Tensor = aten::slice(%7, %13, %11, %240, %13)
                return (%input.13)
            """
    new_g_0 = """
            graph(%input_ids.1, %13, %21, %7, %11):
                %input.13 : Tensor = prim::slice_position_ids(%7, %13, %11, %input_ids.1, %13)
                return (%input.13)
            """
    graph_pairs = [(old_g_0, new_g_0)]
    for g_pair in graph_pairs:
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(g_pair[0], g_pair[1], graph)



def fuse_view(graph):
    old_g = """
            graph(%attn_scores.1, %query1.1, %key2.1, %176, %178, %184, %27, %5, %30, %31):
                %query2.1 : Tensor = aten::view(%query1.1, %176)
                %key3.1 : Tensor = aten::view(%key2.1, %178)
                %182 : Tensor = aten::transpose(%key3.1, %31, %30)
                %attn_scores0.1 : Tensor = aten::baddbmm(%attn_scores.1, %query2.1, %182, %27, %5)
                %attn_scores1.1 : Tensor = aten::view(%attn_scores0.1, %184)
                return (%attn_scores1.1)
            """
    new_g = """
            graph(%attn_scores.1, %query1.1, %key2.1, %176, %178, %184, %27, %5, %30, %31):
                %new30 : int = prim::Constant[value=3]()
                %new31 : int = prim::Constant[value=2]()
                %182 : Tensor = aten::transpose(%key2.1, %new31, %new30)
                %attn_scores0.1 : Tensor = prim::mybaddbmm(%attn_scores.1, %query1.1, %182, %27, %5)
                return (%attn_scores0.1)
            """
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(old_g, new_g, graph)
    torch._C._jit_pass_dce(graph)

def fuse_gather_indices(graph):
    old_g = """
            graph(%input_ids.1, %past_key.1, %30, %31, %32, %26, %27, %33, %28, %29, %34, %35, %36):
                %57 : int = aten::size(%input_ids.1, %32)
                %59 : int = aten::size(%past_key.1, %31)
                %61 : Tensor = aten::add(%57, %59, %32)
                %position_ids.1 : Tensor = aten::arange(%59, %61, %30, %34, %35, %36)
                %64 : Tensor = aten::unsqueeze(%position_ids.1, %33)
                %65 : int[] = prim::ListConstruct(%29, %57)
                %position_ids0.1 : Tensor = aten::view(%64, %65)
                %104 : Tensor = aten::slice(%position_ids0.1, %33, %33, %28, %32)
                %105 : Tensor = aten::unsqueeze(%104, %32)
                %106 : Tensor = aten::slice(%105, %27, %33, %28, %32)
                %gather_indices.1 : Tensor = aten::unsqueeze(%106, %26)
                return (%gather_indices.1)
            """
    new_g = """
            graph(%input_ids.1, %past_key.1, %30, %31, %32, %26, %27, %33, %28, %29, %34, %35, %36):
                %gather_indices0.1 : Tensor = prim::gather_indices(%input_ids.1)
                return (%gather_indices0.1)
            """
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(old_g, new_g, graph)
    torch._C._jit_pass_dce(graph)

class TorchExtractor(object):
    """The TorchExtractor class.

    Decorate the node in model.graph_def, and the new node has the attributes like input_tensors
    and output_tensors, these tensors record the source/dest op name. All of these nodes
    (in a list) will compose a graph, which is Graph class, as the return object.

    Args:
        model: Torchscript Model

    Return:
        Graph: Graph class, the new graph object
    """
    @classmethod
    def __call__(self, model):
        """The __call__ function of the extractor."""
        graph, _ = torch._C._jit_pass_lower_graph(model.graph, model._c)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_remove_inplace_ops(graph)
        torch._C._jit_pass_constant_propagation(graph)
        removeUnusedNode(graph, ['aten::dropout', 'prim::NumToTensor', 'aten::to', 'aten::contiguous',
                                 'aten::alias', 'aten::Int', 'aten::ScalarImplicit'])
        logger.info('Start to extarct torch model ops...')
        fuse_padding_seq(graph)
        fuse_position_ids(graph)
        fuse_view(graph)
        fuse_gather_indices(graph)
        
        new_graph = Graph()
        new_graph.framework_modeling_config['framework'] = 'torch'
        graph_nodes_dict = {}
        util.quant_info_init()

        # insert graph.inputs
        model_input_tensors = []
        for idx, in_tensor in enumerate(graph.inputs()):
            dest_ops = []
            for val_user in in_tensor.uses():
                next_node = val_user.user
                dest_ops.append(get_node_name(next_node))
            dtype = 'fp32'
            if in_tensor.debugName() == 'position_ids':
                continue  # TODO: remove this
            input_tensor = Tensor(name=in_tensor.debugName(),
                source_op=[],
                dest_op=dest_ops,
                shape=[-1, -1],
                data=None,
                dtype=dtype
                )
            graph_nodes_dict[in_tensor.debugName()] = input_tensor
            model_input_tensors.append(input_tensor)
        for idx, out_tensor in enumerate(graph.outputs()):
            new_graph.output_tensors_name.append(out_tensor.debugName())

        # parse weights
        for node in graph.nodes():
            if node.kind() == 'prim::Constant' and node.outputsAt(0).type().kind() == 'TensorType':
                out_val = node.outputsAt(0)
                tensor_name = out_val.debugName()
                out_tensor = out_val.toIValue()
                dest_ops = []
                for val_user in out_val.uses():
                    next_node = val_user.user
                    dest_ops.append(get_node_name(next_node))
                if out_tensor.dtype == torch.qint8:
                    # extrace min max from tensor
                    fp32_data = out_tensor.dequantize()
                    if out_tensor.qscheme() == torch.per_channel_affine or \
                       out_tensor.qscheme() == torch.per_channel_symmetric:
                        # per_channel case
                        per_ch_scale = out_tensor.q_per_channel_scales().numpy()
                        fp32_max = per_ch_scale * 127
                        fp32_min = -fp32_max
                    else:
                        q_scale = torch.tensor(out_tensor.q_scale()).numpy()
                        fp32_max = q_scale * 127
                        fp32_min = -fp32_max
                    dtype = 's8' + "_weight"
                    util.insert_quant_info(tensor_name, [fp32_min, fp32_max, dtype])

                    # ensure weight is sym quantized.
                    out_tensor = out_tensor.int_repr()

                elif out_tensor.dtype == torch.quint8:
                    logger.error("Tensor {} of uint8 is not supported.".format(tensor_name))

                if out_tensor.dtype == torch.float64:
                    fp32_info = torch.finfo(torch.float32)
                    if out_tensor.max().item() <= fp32_info.max and \
                       out_tensor.min().item() >= fp32_info.min:
                        out_tensor = out_tensor.to(torch.float32)
                    else:
                        logger.error("Neural Engine does not support float64 dtype tensor {}."\
                                     .format(tensor_name))
                if out_tensor.dtype == torch.int64:
                    int32_info = torch.iinfo(torch.int32)
                    if out_tensor.max().item() <= int32_info.max and \
                       out_tensor.min().item() >= int32_info.min:
                        out_tensor = out_tensor.to(torch.int32)
                    else:
                        logger.error("Neural Engine does not support int64 dtype tensor {}."\
                                     .format(tensor_name))
                weight = out_tensor.detach().numpy()
                weight_tensor = Tensor(name=tensor_name,
                    source_op=[],
                    dest_op=dest_ops,
                    shape=list(out_tensor.shape),
                    data=weight,
                    dtype=get_data_dtype(weight)
                    )
                graph_nodes_dict[tensor_name] = weight_tensor


        for in_tensor in model_input_tensors:
            if in_tensor.name.split('.')[0] in ['attention_mask', 'position_ids', 'input_ids', 'mask',
                                                'token_type_ids']:
                in_tensor.dtype = 'int32'  # TODO: refine this
        input_data_node = construct_node('input_data',
                                            'Input',
                                            output_tensors=model_input_tensors)
        #new_graph.inputs_dict = graph_nodes_dict
        input_tensors_name = [item for item in graph_nodes_dict]
        new_graph.input_tensors_name = input_tensors_name
        for node in graph.nodes():
            if node.kind() not in op_maps:
                logger.warning("node kind {} is not mapped.".format(node.kind()))
            op_type = op_maps.get(node.kind(), node.kind())
            if op_type not in OPERATORS.keys():
                    op_type = "OpAny"
            new_node = OPERATORS[op_type]()
            new_node.extract('torch', node, model, graph_nodes_dict)
            new_graph.insert_nodes(len(new_graph.nodes), [new_node])

        new_graph.insert_nodes(0, [input_data_node])
        return new_graph
