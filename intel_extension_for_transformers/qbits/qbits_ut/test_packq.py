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
from enum import Enum


def convert_idx(g_idx,  k, blocksize):
    ret_idx = torch.zeros(k, dtype=int)
    g_counter = torch.zeros((k+blocksize-1) // blocksize, dtype=int)
    for i in range(k):
        ret_idx[g_idx[i]*blocksize+g_counter[g_idx[i]]] = i
        g_counter[g_idx[i]] += 1
    return ret_idx


class acquire_type(Enum):
    SIZE = 0
    BLOCKSIZE = 1
    K = 2
    N = 3
    ACT_SHUFFLE = 4
    G_IDX = 5
    WEI_TYPE = 6
    CMPT_TYPE = 7
    SCALE_TYPE = 8
    SCALE_TENSOR = 9
    ZP_TENSOR = 10
    IS_ASYM = 11


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("blocksize", [128])
@pytest.mark.parametrize("compute_type", ["fp32", "bf16", "int8"])
@pytest.mark.parametrize("weight_type", ["int8", "int4_clip"])
@pytest.mark.parametrize("scale_type", ["fp32"])
@pytest.mark.parametrize("asym", [True, False])
def test(m, k, n, weight_type, scale_type, compute_type, asym, blocksize, dump_tensor=False):
    if compute_type == "int8" and asym == True:
        pytest.skip()
    torch.manual_seed(0)
    raw_s8_wei = torch.randint(-128, 127, [k, n], dtype=torch.int8)
    g_idx = torch.arange(k//blocksize, dtype=torch.int)
    g_idx = g_idx.repeat(blocksize)
    cvt_idx = convert_idx(g_idx, k, blocksize)
    zp = torch.randint(-4, 4, [k//blocksize, n], dtype=torch.int8)
    scale = torch.rand(k//blocksize, n, dtype=torch.float)
    packw = qbits.repack_quantized_weight(
        raw_s8_wei, scale, zp, g_idx, weight_type, scale_type, compute_type, asym, blocksize)
    revert_wei = torch.zeros(k, n, dtype=torch.float)
    qbits.dequantize_packed_weight(
        packw, revert_wei, False, compute_type, weight_type, scale_type)
    ref_act = torch.rand(m, k, dtype=torch.float)
    tar_act = ref_act.clone()
    ref_act = torch.index_select(ref_act, 1, cvt_idx)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    qbits.woq_linear(
        tar_act, packw, torch.empty(0), tar_dst, compute_type, weight_type, scale_type, asym)
    ref_dst = torch.matmul(ref_act, revert_wei)
    if dump_tensor:
        print(tar_dst)
        print(ref_dst)
    if compute_type == "fp32":
        assert (abs(ref_dst - tar_dst).max() < 0.03)
    elif compute_type == "bf16":
        assert (abs(ref_dst - tar_dst).max() < 9)
    else:
        assert (abs(ref_dst - tar_dst).max() < 10)
    packw_size = qbits.acquire_packed_weight_info(
        packw, acquire_type.SIZE.value)[0].item()
    if packw_size != packw.size()[0]:
        assert (0)
    packw_wei_type = qbits.acquire_packed_weight_info(
        packw, acquire_type.WEI_TYPE.value)
    packw_wei_type_str = ''.join(chr(ascii_code)
                                 for ascii_code in packw_wei_type.tolist())
    if packw_wei_type_str != weight_type:
        assert (0)
    enable_act_shuffle = qbits.acquire_packed_weight_info(
        packw, acquire_type.ACT_SHUFFLE.value)[0] != 0
    assert (enable_act_shuffle)
    acquire_g_idx = qbits.acquire_packed_weight_info(
        packw, acquire_type.G_IDX.value)
    assert (abs(acquire_g_idx-cvt_idx).max() == 0)
    scale_tensor = qbits.acquire_packed_weight_info(
        packw, acquire_type.SCALE_TENSOR.value)
    assert (abs(scale-scale_tensor).max() == 0)
    is_asym = qbits.acquire_packed_weight_info(
        packw, acquire_type.IS_ASYM.value)[0] != 0
    if is_asym:
        zp_tensor = qbits.acquire_packed_weight_info(
            packw, acquire_type.ZP_TENSOR.value)
        assert (abs(zp-zp_tensor).max() == 0)
