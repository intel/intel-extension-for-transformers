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


def convert_idx(g_idx,  k, group_size):
    ret_idx = torch.zeros(k, dtype=int)
    g_counter = torch.zeros(group_size, dtype=int)
    for i in range(k):
        ret_idx[g_idx[i]*group_size+g_counter[g_idx[i]]] = i
        g_counter[g_idx[i]] += 1
    return ret_idx


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("compute_type", ["fp32", "bf16", "int8"])
@pytest.mark.parametrize("weight_type", ["int8", "int4_clip"])
@pytest.mark.parametrize("scale_type", ["fp32"])
@pytest.mark.parametrize("alg", ["sym", "asym"])
def test(m, k, n, weight_type, scale_type, compute_type, alg, group_size, dump_tensor=False):
    if compute_type == "int8" and alg == "asym":
        pytest.skip()
    torch.manual_seed(0)
    raw_s8_wei = torch.randint(-128, 127, [k, n], dtype=torch.int8)
    g_idx = torch.arange(k//group_size, dtype=torch.int)
    g_idx = g_idx.repeat(group_size)
    cvt_idx = convert_idx(g_idx, k, group_size)
    zp = torch.randint(-4, 4, [k//group_size, n], dtype=torch.int8)
    scale = torch.rand(k//group_size, n, dtype=torch.float)
    packw = torch.ops.jblasop.woq_packq(
        raw_s8_wei, scale, zp, g_idx, weight_type, scale_type, compute_type, alg, group_size)
    revert_wei = torch.zeros(k, n, dtype=torch.float)
    torch.ops.jblasop.woq_dequantize(
        packw, revert_wei, False, compute_type, weight_type, scale_type)
    ref_act = torch.rand(m, k, dtype=torch.float)
    tar_act = ref_act.clone()
    ref_act = torch.index_select(ref_act, 1, cvt_idx)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    torch.ops.jblasop.woq_linear(
        tar_act, packw, torch.empty(0), tar_dst, n, False, compute_type, weight_type, scale_type, alg == "asym")
    ref_dst = torch.matmul(ref_act, revert_wei)
    if dump_tensor:
        print(tar_dst)
        print(ref_dst)
    if compute_type == "fp32":
        assert (abs(ref_dst - tar_dst).max() < 0.03)
    elif compute_type == "bf16":
        assert (abs(ref_dst - tar_dst).max() < 8)
    else:
        assert (abs(ref_dst - tar_dst).max() < 10)
