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


import torch
import intel_extension_for_pytorch
import intel_extension_for_transformers.gbits as gbits
from functools import reduce
from operator import mul
from ...utils import logger

class ParamsGBits(torch.nn.Parameter):
    def __new__(
            cls,
            data=None,
            requires_grad=True,
            quant_state=None,
            blocksize=32,
            compress_statistics=True,
            quant_dtype=None,
            scale_dtype="fp32",
    ):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_dtype = quant_dtype
        self.scale_dtype = scale_dtype
        self.quant_state = quant_state
        self.data = data
        return self


class QuantizedLinearGPU(torch.nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype="fp32",
        compress_statistics=True,
        weight_dtype='int4_fullrange',
        scale_dtype='fp32',
        blocksize=32,
        scheme="sym",
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        assert compute_dtype in ['fp32', 'fp16'], \
            "compute_dtype must be 'fp32', 'fp16' on intel CPU device."
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = blocksize
        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype
        self.device = device

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        shape = list(x.size())
        m = reduce(mul, shape[0:-1])
        out = torch.empty(m, self.out_features, dtype=x.dtype, device=self.device)
        bias = torch.zeros(0) if self.bias is None else self.bias.data
        gbits.linear(
            x.view(m, shape[-1]), self.weight.data, bias, out,
            self.out_features, self.bias is not None, self.compute_dtype, self.weight_dtype, self.scale_dtype, self.self.blocksize)
        shape[-1] = self.out_features
        out = out.view(shape)
        return out

    def init_weights_bias(self, weight_data, bias=None):
        weight = gbits.quantize(
            weight_data, True, self.blocksize, self.compute_dtype, self.weight_dtype, self.scale_dtype
        )
        self.weight = ParamsGBits(
            data=weight, requires_grad=False, quant_state={"scheme": self.scheme}, blocksize=self.blocksize,
            compress_statistics=self.compress_statistics, quant_dtype=self.weight_dtype, scale_dtype=self.scale_dtype
        )
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
