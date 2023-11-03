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

class MatMul4Bit(torch.autograd.Function):
    # forward is the same, but we added the fallback for pre-turing GPUs
    # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=None):
        # default of pytorch behavior if inputs are empty
        ctx.is_empty = False
        if prod(A.shape) == 0:
            ctx.is_empty = True
            ctx.A = A
            ctx.B = B
            ctx.bias = bias
            B_shape = state[1]
            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)


        # 1. Dequantize
        # 2. MatmulnN
        # torch.ops.weight_only_jblasop.jblas_symqdq_weight(B, False, 4, 32) # TODO: replace with dequantize
        output = torch.nn.functional.linear(A, B.to(A.dtype), bias)

        # 3. Save state
        ctx.state = state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

        if any(ctx.needs_input_grad[:2]):
            ctx.tensors = (A, B)
        else:
            ctx.tensors = (None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_empty:
            bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
            return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

        req_gradA, _, _, req_gradBias, _= ctx.needs_input_grad
        A, B = ctx.tensors
        state = ctx.state

        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

        # not supported by PyTorch. TODO: create work-around
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        # torch.ops.weight_only_jblasop.jblas_symqdq_weight(B, False, 4, 32) # TODO: replace with dequantize
        if req_gradA: grad_A = torch.matmul(grad_output, B.to(grad_output.dtype))

        return grad_A, grad_B, None, grad_bias, None

def matmul_4bit(A: Tensor, B: Tensor, quant_state: List = None, out: Tensor = None, bias=None, do_dequant=True):
    # assert quant_state is not None
    if do_dequant:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state) # TODO: replace with 4bit matmul