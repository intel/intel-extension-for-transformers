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


import os
import operator
import torch
from functools import reduce
from torch import Tensor
from typing import Tuple, Optional, List
from enum import Enum


class qbits_acquire_type(Enum):
    SIZE = 0
    BLOCKSIZE = 1
    K = 2
    N = 3
    ACT_SHUFFLE = 4
    G_IDX = 5
    WEI_TYPE = 6
    CMPT_TYPE = 7
    SCALE_TYPE = 8


def qbits_woq_linear_ref_impl(activation, packw,  bias, compute_type, weight_type, scale_type):
    assert (activation.is_contiguous())
    assert (packw.is_contiguous())
    activation = activation.to(torch.float32)
    n = torch.ops.bestlaop.acquire_woq_packw_info(
        packw, qbits_acquire_type.N.value)[0].item()
    k = activation.shape[1]
    revert_wei = torch.empty(k, n, dtype=torch.float)
    torch.ops.bestlaop.woq_dequantize(
        packw, revert_wei, False, compute_type, weight_type, scale_type)
    enable_act_shuffle = torch.ops.bestlaop.acquire_woq_packw_info(
        packw, qbits_acquire_type.ACT_SHUFFLE.value)[0] != 0
    if enable_act_shuffle:
        g_idx = torch.ops.bestlaop.acquire_woq_packw_info(
            packw, qbits_acquire_type.G_IDX.value)
        activation = torch.index_select(activation, 1, g_idx)
    out = torch.matmul(activation, revert_wei)
    if bias is not None:
        assert (bias.is_contiguous())
        assert (bias.dtype == torch.float32)
        out += bias
    return out


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class MatMulKBit(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A,
        B,
        out=None,
        bias=None,
        compute_dtype=None,
        weight_dtype=None,
        scale_dtype=None,
        scheme=None,
    ):
        # # 1. Dequantize
        # B_dequant = torch.zeros(out.shape[-1], A.shape[-1], dtype=torch.float)
        # torch.ops.bestlaop.woq_dequantize(
        #     B, B_dequant, True, compute_dtype, weight_dtype, scale_dtype)
        # B_dequant = B_dequant.to(dtype=A.dtype)

        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B  # B_dequant
            ctx.bias = bias
            B_shape = (out.shape[-1], A.shape[-1])  # B_dequant.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(
                    A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device
                )
            else:
                return torch.empty(
                    A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device
                )

        # 2. Matmul
        # output = torch.nn.functional.linear(A, B_dequant, bias)
        qbits_debug_flag = os.getenv('QBITS_DEBUG', 'NULL')
        if qbits_debug_flag == 'NULL':
            torch.ops.bestlaop.woq_linear(
                A,
                B.data,
                bias,
                out,
                out.shape[-1],
                bias is not None,
                compute_dtype,
                weight_dtype,
                scale_dtype,
                False if scheme == "sym" else True,
            )
        else:
            out = qbits_woq_linear_ref_impl(
                A, B.data, bias, compute_dtype, weight_dtype, scale_dtype)
        output = out

        # 3. Save state
        ctx.compute_dtype, ctx.weight_dtype, ctx.scale_dtype = (
            compute_dtype,
            weight_dtype,
            scale_dtype,
        )
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = (
            A.dtype,
            B.dtype,
            None if bias is None else bias.dtype,
        )
        # B_dequant.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B)  # B_dequant
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(
                ctx.bias)
            return (
                torch.zeros_like(ctx.A),
                torch.zeros_like(ctx.B),
                None,
                bias_grad,
                None,
            )

        req_gradA, _, _, req_gradBias, _, _, _, _ = ctx.needs_input_grad
        A, B = ctx.tensors
        grad_A, grad_B, grad_bias = None, None, None

        B_dequant = torch.zeros(
            grad_output.shape[-1], A.shape[-1], dtype=torch.float)

        torch.ops.bestlaop.woq_dequantize(
            B, B_dequant, True, ctx.compute_dtype, ctx.weight_dtype, ctx.scale_dtype
        )

        B = B_dequant

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA:
            grad_A = torch.matmul(grad_output, B.to(grad_output.dtype))

        return grad_A, grad_B, None, grad_bias, None, None, None, None


def matmul_kbit(
    A: Tensor,
    B: Tensor,
    bias,
    out,
    compute_dtype,
    weight_dtype,
    scale_dtype,
    scheme,
    do_dequant=False,
):

    if do_dequant:
        return MatMulKBit.apply(
            A, B, out, bias, compute_dtype, weight_dtype, scale_dtype, scheme
        )
    else:
        qbits_debug_flag = os.getenv('QBITS_DEBUG', 'NULL')
        if qbits_debug_flag == 'NULL':
            torch.ops.bestlaop.woq_linear(
                A,
                B.data,
                bias,
                out,
                out.shape[-1],
                bias is not None,
                compute_dtype,
                weight_dtype,
                scale_dtype,
                False if scheme == "sym" else True,
            )
        else:
            out = qbits_woq_linear_ref_impl(
                A, B.data, bias, compute_dtype, weight_dtype, scale_dtype)

        return out
