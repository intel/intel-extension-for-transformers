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

from ut_utils import *

configs = {"s8_scalef32": {"int8", "fp32"}, "s4clip_scalef32": {"int8", "fp32", "bf16"}, "s4fullrange_scalef32": {
    "int8", "fp32", "bf16"}, "fp4bnb_scalef32": {"fp32", "bf16"}, "fp4e2m1_scalef32": {"fp32", "bf16"}, "nf4_scalef32": {"fp32", "bf16"}}


@capture_args
@pytest.mark.parametrize("m", (256,))
@pytest.mark.parametrize("n", (1024,))
@pytest.mark.parametrize("k", (512,))
@pytest.mark.parametrize("blocksize", (128, -1))
@pytest.mark.parametrize("compute_type", ["int8", "fp32", "bf16"])
@pytest.mark.parametrize("weight_type", ["s8_scalef32", "s4clip_scalef32", "s4fullrange_scalef32", "nf4_scalef32", "fp4bnb_scalef32", "fp4e2m1_scalef32"])
@pytest.mark.parametrize("transpose", (True, False))
@pytest.mark.parametrize("add_bias", (True, False))
@pytest.mark.parametrize("dt", ("fp32", "bf16"))
def test(m, n, k, blocksize, compute_type, weight_type, transpose, add_bias, dt, dump_tensor_info=True):
    if compute_type not in configs[weight_type]:
        pytest.skip()
    torch.manual_seed(0)
    ref_activation = torch.rand(m, k, dtype=torch.float)
    tar_activation = ref_activation.clone()
    if dt == "bf16":
        tar_activation = ref_activation.to(torch.bfloat16)
    wei_row = k
    wei_col = n
    if transpose:
        wei_row, wei_col = wei_col, wei_row
    raw_wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = torch.ops.jblasop.woq_quantize(
        raw_wei, transpose, blocksize, compute_type, weight_type)
    revert_wei = torch.zeros(wei_row, wei_col, dtype=torch.float)
    torch.ops.jblasop.woq_dequantize(
        compress_wei, revert_wei, transpose, compute_type, weight_type)
    bias = torch.rand(n, dtype=torch.float)*10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if dt == "bf16":
        tar_dst = tar_dst.to(torch.bfloat16)
    if transpose:
        revert_wei = torch.transpose(revert_wei, 0, 1)
    ref_dst = torch.matmul(ref_activation, revert_wei)
    torch.ops.jblasop.woq_linear(
        tar_activation, compress_wei, bias, tar_dst, n, add_bias, compute_type, weight_type)
    if dt == "bf16":
        tar_dst = tar_dst.to(torch.float)
    if add_bias:
        ref_dst += bias
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    assert (torch.allclose(tar_dst, ref_dst, rtol=0.03))
