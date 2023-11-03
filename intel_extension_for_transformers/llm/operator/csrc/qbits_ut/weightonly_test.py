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
import inspect
from functools import wraps
torch.ops.load_library("../build/libqbits.so")


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
def test(m, n, k, blocksize, compute_type, weight_type, transpose, add_bias, src_dt, dst_dt, dump_tensor_info=False):
    torch.manual_seed(0)
    ref_activation = torch.rand(m, k, dtype=torch.float)
    tar_activation = ref_activation.clone()
    if src_dt == "bf16":
        tar_activation = ref_activation.to(torch.bfloat16)
    wei_row = k
    wei_col = n
    if transpose:
        wei_row, wei_col = wei_col, wei_row
    raw_wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = torch.ops.weight_only_jblasop.qbits_quantize(
        raw_wei, transpose, blocksize, compute_type, weight_type)
    revert_wei = torch.zeros(wei_row, wei_col, dtype=torch.float)
    torch.ops.weight_only_jblasop.qbits_dequantize(
        compress_wei, revert_wei, transpose, compute_type, weight_type)
    bias = torch.rand(n, dtype=torch.float)*10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if dst_dt == "bf16":
        tar_dst = tar_dst.to(torch.bfloat16)
    if transpose:
        revert_wei = torch.transpose(revert_wei, 0, 1)
    ref_dst = torch.matmul(ref_activation, revert_wei)
    torch.ops.weight_only_jblasop.qbits_linear(
        tar_activation, compress_wei, bias, tar_dst, n, add_bias, compute_type, weight_type)
    if dst_dt == "bf16":
        tar_dst = tar_dst.to(torch.float)
    if add_bias:
        ref_dst += bias
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    if torch.allclose(tar_dst, ref_dst, rtol=0.03):
        print("ok")
    else:
        print("fail")


configs = {"s8_scalef32": {"int8", "fp32"}, "s4clip_scalef32": {"int8", "fp32", "bf16"}, "s4fullrange_scalef32": {
    "int8", "fp32", "bf16"}, "fp4bnb_scalef32": {"fp32", "bf16"}, "fp4e2m1_scalef32": {"fp32", "bf16"}, "nf4_scalef32": {"fp32", "bf16"}}

blocksizes = [128, -1]
do_trans = [False, True]
add_bias = [False, True]
src_dts = ["fp32", "bf16"]
dst_dts = ["fp32", "bf16"]

workspace = torch.zeros(786432, dtype=torch.int8)
torch.ops.weight_only_jblasop.qbits_set_weightonly_workspace(workspace)

for weight_type in configs:
    m = 256
    n = 1024
    k = 512  # contain unalign calc error bug currently.
    for compute_type in configs[weight_type]:
        for blocksize in blocksizes:
            if compute_type == "int8" and blocksize % 8 != 0 and blocksize != -1:
                continue
            if blocksize == -1:
                if weight_type != "s8_scalef32" or compute_type != "int8":
                    continue
            for trans in do_trans:
                for bias in add_bias:
                    for src_dt in src_dts:
                        for dst_dt in dst_dts:
                            test(m, n, k, blocksize, compute_type,
                                 weight_type, trans, bias, src_dt, dst_dt)
