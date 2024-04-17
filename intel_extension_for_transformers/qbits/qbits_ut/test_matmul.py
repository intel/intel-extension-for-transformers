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


@capture_args
@pytest.mark.parametrize("m", (256, ))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("k", (1024,))
@pytest.mark.parametrize("trans_matB", (True, False))
@pytest.mark.parametrize("dt", ("fp32", "bf16"))
def test(m, n, k, trans_matB, dt, dump_tensor_info=True):
    torch.manual_seed(0)
    activation = torch.rand(m, k, dtype=torch.float)
    activation_cp = activation.clone()
    if dt == "bf16":
        activation = activation.to(torch.bfloat16)
    wei_row = k
    wei_col = n
    if trans_matB:
        wei_row, wei_col = wei_col, wei_row
    wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    cp_wei = wei.clone()
    if dump_tensor_info:
        print(wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if dt == "bf16":
        tar_dst = tar_dst.to(torch.bfloat16)
        wei = wei.to(torch.bfloat16)
    qbits.matmul(activation, wei, tar_dst, trans_matB)
    if trans_matB:
        cp_wei = torch.transpose(cp_wei, 0, 1)
    ref_dst = torch.matmul(activation_cp, cp_wei)
    if dt == "bf16":
        tar_dst = tar_dst.to(torch.float)
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    assert (torch.allclose(tar_dst, ref_dst, rtol=0.03))
