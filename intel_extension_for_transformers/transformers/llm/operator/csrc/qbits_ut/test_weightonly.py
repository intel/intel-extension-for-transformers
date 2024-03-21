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

cmpt_configs = {"int8": {"int8", "bf16", "fp32"}, "int4_clip": {"int8", "fp32", "bf16"}, "int4_fullrange": {
    "int8", "fp32", "bf16"}, "fp4_e2m1_bnb": {"fp32", "bf16"}, "fp4_e2m1": {"fp32", "bf16"}, "nf4": {"fp32", "bf16"},
    "fp8_e5m2": {"fp32", "bf16"}, "fp8_e4m3": {"fp32", "bf16"}
}

scale_configs = {"int8": {"fp32", "bf16"}, "int4_clip": {"fp32", "bf16"}, "int4_fullrange": {"fp32", "bf16"}, "fp4_e2m1_bnb": {"fp32", "bf16"}, "fp4_e2m1": {"fp32", "bf16"}, "nf4": {"fp32", "bf16"},
                 "fp8_e5m2": {"fp32", "fp8_e8m0"}, "fp8_e4m3": {"fp32", "fp8_e8m0"}}

asym_configs = {"int8", "int4_clip", "int4_fullrange"}


@capture_args
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("blocksize", [128, -1])
@pytest.mark.parametrize("compute_type", ["int8", "bf16", "fp32"])
@pytest.mark.parametrize("weight_type", ["int8", "int4_clip", "int4_fullrange", "nf4", "fp4_e2m1_bnb", "fp4_e2m1", "fp8_e5m2", "fp8_e4m3"])
@pytest.mark.parametrize("scale_type", ["fp32", "bf16", "fp8_e8m0"])
@pytest.mark.parametrize("asym", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("add_bias", [True, False])
@pytest.mark.parametrize("src_dt", ["fp32", "bf16"])
@pytest.mark.parametrize("dst_dt", ["fp32", "bf16"])
def test(m, n, k, blocksize, compute_type, weight_type, scale_type, asym, transpose, add_bias, src_dt, dst_dt, dump_tensor_info=True):
    if compute_type not in cmpt_configs[weight_type] or scale_type not in scale_configs[weight_type]:
        pytest.skip()
    if asym and (weight_type not in asym_configs or compute_type == "int8" or scale_type != "fp32"):
        pytest.skip()
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
    compress_wei = qbits.quantize_to_packed_weight(
        raw_wei, transpose, blocksize, compute_type, weight_type, scale_type, asym)
    revert_wei = torch.zeros(wei_row, wei_col, dtype=torch.float)
    qbits.dequantize_packed_weight(
        compress_wei, revert_wei, transpose, compute_type, weight_type, scale_type)
    bias = torch.empty(0)
    if add_bias:
        bias = torch.rand(n, dtype=torch.float)*10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if dst_dt == "bf16":
        tar_dst = tar_dst.to(torch.bfloat16)
    if transpose:
        revert_wei = torch.transpose(revert_wei, 0, 1)
    ref_dst = torch.matmul(ref_activation, revert_wei)
    qbits.woq_linear(
        tar_activation, compress_wei, bias, tar_dst, compute_type, weight_type, scale_type, asym)
    if dst_dt == "bf16":
        tar_dst = tar_dst.to(torch.float)
    if add_bias:
        ref_dst += bias
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    assert (torch.allclose(tar_dst, ref_dst, rtol=0.03))
