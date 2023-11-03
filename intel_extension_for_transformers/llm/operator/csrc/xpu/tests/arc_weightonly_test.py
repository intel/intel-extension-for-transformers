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
import gbits_linear
import gbits_quantize
import gbits_dequantize
import inspect
from functools import wraps


def capture_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_strs = []
        for name, value in bound_args.arguments.items():
            arg_strs.append(f'{name}={value}')
        result = ', '.join(arg_strs)
        print(result)
        return f(*args, **kwargs)
    return wrapper

@capture_args
def test(m, n, k, blocksize, compute_type, weight_type, transpose, add_bias, dump_tensor_info=False):
    torch.manual_seed(0)
    ref_activation = torch.rand(m, k, dtype=torch.float)
    tar_activation = ref_activation.clone()
    if compute_type == "fp16":
        tar_activation = ref_activation.to(torch.float16)
    wei_row = k
    wei_col = n
    if transpose:
        wei_row, wei_col = wei_col, wei_row
    raw_wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = gbits_quantize.forward(
        raw_wei, transpose, blocksize, compute_type, weight_type)
    revert_wei = torch.zeros(wei_row, wei_col, dtype=torch.float)
    gbits_dequantize.forward(
        compress_wei, revert_wei, transpose, compute_type, weight_type)
    bias = torch.rand(n, dtype=torch.float)*10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if compute_type == "fp16":
        tar_dst = tar_dst.to(torch.float16)
    if transpose:
        revert_wei = torch.transpose(revert_wei, 0, 1)
    ref_dst = torch.matmul(ref_activation, revert_wei)
    gbits_linear.forward(
        tar_activation, compress_wei, bias, tar_dst, n, add_bias, compute_type, weight_type)
    tar_dst = tar_dst.to(torch.float)
    if add_bias:
        ref_dst += bias
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    if torch.allclose(tar_dst, ref_dst, rtol=0.03):
        print("ok")
    else:
        print(torch.max(torch.abs(tar_dst - ref_dst)))
        print("fail")

configs = {"s4fullrange_scalef32": {"fp32", "fp16"}}
blocksizes = [128]
do_trans = [False, True]
add_bias = [False, True]

for weight_type in configs:
    m = 256
    n = 1024
    k = 512
    for compute_type in configs[weight_type]:
        for blocksize in blocksizes:
            for trans in do_trans:
                for bias in add_bias:
                    test(m, n, k, blocksize, compute_type,
                         weight_type, trans, bias)
