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


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("group_size", [128])
# @pytest.mark.parametrize("compute_type", ["int8", "bf16", "fp32"])
@pytest.mark.parametrize("compute_type", ["fp32"])
# @pytest.mark.parametrize("weight_type", ["int8", "int4_clip"])
@pytest.mark.parametrize("weight_type", ["int8"])
@pytest.mark.parametrize("scale_type", ["fp32"])
@pytest.mark.parametrize("alg", ["sym"])
def test(m, k, n, weight_type, scale_type, compute_type, alg, group_size):
    torch.manual_seed(0)
    raw_s8_wei = torch.randint(-128, 127, [k, n], dtype=torch.int8)
    # g_idx = torch.randperm(k, dtype=torch.int)
    g_idx = torch.empty(0)
    zp = torch.empty(0)
    scale = torch.rand(k//group_size, n, dtype=torch.float)
    packw = torch.ops.jblasop.woq_packq(
        raw_s8_wei, scale, zp, g_idx, weight_type, scale_type, compute_type, alg, group_size)
    revert_wei = torch.zeros(k, n, dtype=torch.float)
    torch.ops.jblasop.woq_dequantize(
        packw, revert_wei, False, compute_type, weight_type, scale_type)
    ref_act = torch.rand(m, k, dtype=torch.float)
    tar_act = ref_act.clone()
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    torch.ops.jblasop.woq_linear(
        ref_act, packw, torch.empty(0), tar_dst, n, False, compute_type, weight_type, scale_type, alg == "asym")
    print(tar_dst)
