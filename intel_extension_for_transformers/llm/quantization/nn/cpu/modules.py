#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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


import os
import torch
from functools import reduce
from operator import mul


torch.ops.load_library(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../..", "libqbits.so")
)


class ParamsQBits(torch.nn.Parameter):
    def __new__(
            cls,
            data=None,
            requires_grad=True,
            quant_state=None,
            blocksize=32,
            compress_statistics=True,
            quant_dtype='int8'
    ):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_dtype = quant_dtype
        self.quant_state = quant_state
        self.data = data
        return self

# class Params4Bits(torch.nn.Parameter):
    # def __new__(
    #         cls,
    #         data=None,
    #         requires_grad=True,
    #         quant_state=None,
    #         blocksize=32,
    #         compress_statistics=True,
    #         quant_dtype='nf4'
    # ):
    #     if data is None:
    #         data = torch.empty(0)

    #     self = torch.Tensor._make_subclass(cls, data, requires_grad)
    #     self.blocksize = blocksize
    #     self.compress_statistics = compress_statistics
    #     self.quant_dtype = quant_dtype
    #     self.quant_state = quant_state
    #     self.data = data
    #     return self


class QuantizedLinearCPU(torch.nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype="fp32",
        compress_statistics=True,
        weight_dtype='s4fullrange_scalef32',
        blocksize=32,
        scheme="sym",
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        assert compute_dtype in ['fp32', 'bf16', 'int8'], \
            "compute_dtype must be 'fp32', 'bf16', 'int8' on intel CPU device."
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = blocksize
        self.scheme = scheme
        self.weight_dtype = weight_dtype

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, 'quant_state', None) is None:
            print('quantization state not initialized. Please call .init_weights_bias().')

        shape = list(x.size())
        m = reduce(mul, shape[0:-1])
        out = torch.zeros(m, self.out_features, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.data
        torch.ops.weight_only_jblasop.qbits_linear(
            x.view(m, shape[-1]), self.weight.data, bias, out,
            self.out_features, self.bias is not None, self.compute_dtype, self.weight_dtype
        )
        shape[-1] = self.out_features
        out = out.view(shape)

        return out

    def init_weights_bias(self, weight_data, bias=None):
        shape = weight_data.shape
        weight = torch.ops.weight_only_jblasop.qbits_quantize(
            weight_data, True, self.blocksize, self.compute_dtype, self.weight_dtype)
        weight.resize_(shape)
        self.weight = ParamsQBits(
            data=weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
            compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype
        )
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)


# class QuantizedLinearINT4(QuantizedLinearCPU, LoraLayer):
#     def __init__(
#         self,
#         input_features,
#         output_features,
#         bias=True,
#         compute_dtype="fp32",
#         compress_statistics=True,
#         weight_dtype="s4fullrange_scalef32",
#         blocksize=32,
#         scheme="sym",
#         device=None,
#         use_lora=True,
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.01,
#     ):
#         QuantizedLinearCPU.__init__(self, input_features, output_features, bias, compute_dtype, compress_statistics,
#                                       weight_dtype, blocksize, scheme, device)
#         self.use_lora = use_lora
#         if self.use_lora:
#             LoraLayer.__init__(self, input_features, output_features)
#             self._r = r
#             self._lora_alpha = lora_alpha
#             self._lora_dropout = lora_dropout

#     def forward(self, x: torch.Tensor):
#         if self.bias is not None and self.bias.dtype != x.dtype:
#             self.bias.data = self.bias.data.to(x.dtype)

#         # if getattr(self.weight, 'quant_state', None) is None:
#         #     print('FP4 quantization state not initialized. Please call .quantize_weights().')
#         inp_dtype = x.dtype
#         if self.compute_dtype is not None:
#             x = x.to(self.compute_dtype)

#         bias = None if self.bias is None else self.bias.to(self.compute_dtype)
#         out = matmul_4bit(x, self.weight, bias=bias, quant_state=self.weight.quant_state)
#         x = x.to(self.lora_A[self.active_adapter].weight.dtype)

#         if self.use_lora:
#             out += (
#                 self.lora_B[self.active_adapter](
#                     self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
#                 )
#                 * self.scaling[self.active_adapter]
#             )

#         return out

#     def init_weights_bias(self, weight_data, bias=None):
#         weight = torch.ops.weight_only_jblasop.qbits_quantize(
#             weight_data, True, self.blocksize, self.compute_dtype, self.weight_dtype)
#         self.weight = ParamsQBits(
#             data=weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
#             compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype
#         )
#         self.weight.requires_grad = False
#         if bias is not None:
#             self.bias = torch.nn.Parameter(bias, requires_grad=False)

#         if self.use_lora:
#             self.active_adapter = 'default'
#             LoraLayer.update_layer(self, self.active_adapter, self._r, self._lora_alpha, self._lora_dropout, True)

# class QuantizedLinearINT8(QuantizedLinearCPU):
#     def __init__(
#         self,
#         input_features,
#         output_features,
#         bias=True,
#         compute_dtype="fp32",
#         scale_dtype="fp32",
#         compress_statistics=True,
#         blocksize=32,
#         scheme="sym",
#         device=None,
#     ):
#         super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics,
#                          "s8_scalef32", blocksize, scheme, device)
