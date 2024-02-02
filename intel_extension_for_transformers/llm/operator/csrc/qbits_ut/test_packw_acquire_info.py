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


class acquire_type(Enum):
    SIZE = 0
    BLOCKSIZE = 1
    K = 2
    N = 3
    TRANSPOSE = 4
    ACT_ORDER = 5
    G_IDX = 6
    WEI_TYPE = 7
    CMPT_TYPE = 8
    SCALE_TYPE = 9


@capture_args
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [512])
@pytest.mark.parametrize("blocksize", [128])
@pytest.mark.parametrize("compute_type", ["fp32"])
@pytest.mark.parametrize("weight_type", ["int4_clip"])
@pytest.mark.parametrize("scale_type", ["fp32"])
@pytest.mark.parametrize("asym", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test(n, k, blocksize, compute_type, weight_type, scale_type, asym, transpose,  dump_tensor_info=True):
    torch.manual_seed(0)
    wei_row = k
    wei_col = n
    if transpose:
        wei_row, wei_col = wei_col, wei_row
    raw_wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = torch.ops.bestlaop.woq_quantize(
        raw_wei, transpose, blocksize, compute_type, weight_type, scale_type, asym)
    compress_wei_size = torch.ops.bestlaop.acquire_woq_packw_info(
        compress_wei, acquire_type.SIZE.value)
    if compress_wei_size[0].item() != compress_wei.size()[0]:
        print(compress_wei_size[0].item())
        print(compress_wei.size()[0])
        assert (0)
