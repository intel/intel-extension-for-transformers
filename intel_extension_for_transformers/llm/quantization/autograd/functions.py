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


import operator
import torch
from functools import reduce
from torch import Tensor
from typing import Tuple, Optional, List

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class MatMulKBit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, compute_dtype=None, weight_dtype=None):
        # # 1. Dequantize
        # B_dequant = torch.zeros(out.shape[-1], A.shape[-1], dtype=torch.float)
        # torch.ops.jblasop.woq_dequantize(
        #     B, B_dequant, True, compute_dtype, weight_dtype)
        # B_dequant = B_dequant.to(dtype=A.dtype)

        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B # B_dequant
            ctx.bias = bias
            B_shape = (out.shape[-1], A.shape[-1]) # B_dequant.shape
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 2. Matmul
        # output = torch.nn.functional.linear(A, B_dequant, bias)
        torch.ops.jblasop.woq_linear(
            A, B.data, bias, out, out.shape[-1], bias is not None, compute_dtype, weight_dtype
        )
        output = out

        # 3. Save state
        ctx.compute_dtype, ctx.weight_dtype = compute_dtype, weight_dtype
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype
        # B_dequant.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B) # B_dequant
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, _, _, req_gradBias, _, _ = ctx.needs_input_grad
        A, B = ctx.tensors
        grad_A, grad_B, grad_bias = None, None, None

        B_dequant = torch.zeros(grad_output.shape[-1], A.shape[-1], dtype=torch.float)
        torch.ops.jblasop.woq_dequantize(
            B, B_dequant, True, ctx.compute_dtype, ctx.weight_dtype)
        B = B_dequant

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA: grad_A = torch.matmul(grad_output, B.to(grad_output.dtype))

        return grad_A, grad_B, None, grad_bias, None, None

def matmul_kbit(A: Tensor, B: Tensor, bias, out, compute_dtype, weight_dtype, do_dequant=False):
    if do_dequant:
        return MatMulKBit.apply(A, B, out, bias, compute_dtype, weight_dtype)
    else:
        torch.ops.jblasop.woq_linear(
            A, B.data, bias, out, out.shape[-1], bias is not None, compute_dtype, weight_dtype
        )
        return out