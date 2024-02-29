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
import torch
from functools import reduce
from operator import mul
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING, PeftType
from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils.other import transpose
from intel_extension_for_transformers.llm.quantization.autograd import matmul_kbit


torch.ops.load_library(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "libqbits.so")
)


class DropoutQBits_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, probability):
        mask = torch.ops.qbits_customop.dropout_fwd(input, probability)
        if any(ctx.needs_input_grad[:1]):
            ctx.tensors = (mask,)
        else:
            ctx.tensors = (None,)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        req_grad_input, _ = ctx.needs_input_grad
        mask = ctx.tensors[0]
        grad_input = None

        if req_grad_input:
            grad_input = torch.ops.qbits_customop.dropout_bwd(grad_output, mask)

        return grad_input, None


class DropoutQBits(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return DropoutQBits_.apply(input, self.p)
        else:
            return input


class ParamsQBits(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=True,
        quant_state=None,
        blocksize=32,
        compress_statistics=True,
        quant_dtype=None,
        scale_dtype="fp32",
    ):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_dtype = quant_dtype
        self.scale_dtype = scale_dtype
        self.quant_state = quant_state
        self.data = data
        return self


class QuantizedLinearQBits(torch.nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype="fp32",
        compress_statistics=True,
        weight_dtype="int4_clip",
        scale_dtype="fp32",
        blocksize=32,
        scheme="sym",
        device=None,
    ):
        super().__init__(input_features, output_features, bias, device)
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = blocksize
        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        shape = list(x.size())
        m = reduce(mul, shape[0:-1])
        out = torch.zeros(m, self.out_features, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.data
        if not x.is_contiguous():
            x = x.contiguous()
        out = matmul_kbit(
            x.view(m, shape[-1]),
            self.weight,
            bias,
            out,
            self.compute_dtype if self.compute_dtype is not None else "fp32",
            self.weight_dtype,
            self.scale_dtype if self.scale_dtype is not None else "fp32",
            self.scheme,
            do_dequant=self.training,
        )
        shape[-1] = self.out_features
        out = out.view(shape)

        return out

    def set_weights_bias(self, weight_data, bias=None):
        if weight_data.is_meta:
            weight_data = torch.ones(weight_data.shape, dtype=torch.float)
        weight = torch.ops.bestlaop.woq_quantize(
            weight_data,
            True,
            self.blocksize,
            self.compute_dtype if self.compute_dtype is not None else "fp32",
            self.weight_dtype,
            self.scale_dtype if self.scale_dtype is not None else "fp32",
            False if self.scheme == "sym" else True,
        )
        self.weight = ParamsQBits(
            data=weight,
            requires_grad=False,
            quant_state={"scheme": self.scheme},
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_dtype=self.weight_dtype,
            scale_dtype=self.scale_dtype,
        )
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

    def set_gptq_weights_bias(
        self,
        int_weight,
        gptq_scales,
        gptq_zeros,
        g_idx,
        q_config,
        bias=None,
    ):

        if q_config.gptq_quantize_config["desc_act"]:
            int_weight2 = int_weight.clone()
            group_size = q_config.gptq_quantize_config["group_size"]
            group_dict = {}
            for i in range(len(g_idx)):
                group_idx = g_idx[i].item()
                if group_idx not in group_dict:
                    target_idx = group_idx * group_size
                    group_dict[group_idx] = 0
                else:
                    group_dict[group_idx] = group_dict[group_idx] + 1
                    target_idx = group_idx * group_size + group_dict[group_idx]
                int_weight2[target_idx] = int_weight[i]
            int_weight = int_weight2

        if q_config.gptq_quantize_config["bits"] == 4:
            int_weight = (int_weight - 8) * 16
            gptq_scales = gptq_scales / 16
            gptq_zeros = (gptq_zeros - 8) * 16
        if q_config.gptq_quantize_config["sym"]:
            gptq_zeros = torch.empty(0, dtype=torch.int8)

        if not q_config.gptq_quantize_config["desc_act"]:
            g_idx = torch.empty(0, dtype=torch.int32)

        packw = torch.ops.bestlaop.woq_packq(
            int_weight.contiguous(),
            gptq_scales.float().contiguous(),
            gptq_zeros.contiguous(),
            g_idx.contiguous(),
            q_config.weight_dtype,
            q_config.scale_dtype,
            q_config.compute_dtype,
            not q_config.gptq_quantize_config["sym"],
            self.blocksize,
        )

        self.weight = ParamsQBits(
            data=packw,
            requires_grad=False,
            quant_state={"scheme": self.scheme},
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_dtype=self.weight_dtype,
            scale_dtype=self.scale_dtype,
        )
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)


class QuantizedLoraLinearQBits(QuantizedLinearQBits, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        QuantizedLinearQBits.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            compute_dtype=kwargs.get("compute_dtype", "fp32"),
            compress_statistics=kwargs.get("compress_statistics", True),
            weight_dtype=kwargs.get("weight_dtype", "int4_fullrange"),
            scale_dtype=kwargs.get("scale_dtype", "fp32"),
            blocksize=kwargs.get("blocksize", 32),
            scheme=kwargs.get("scheme", "sym"),
            device=kwargs.get("device", None),
        )
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        qbits_customop_available = True
        try:
            torch.ops.qbits_customop.dropout_fwd
        except:
            qbits_customop_available = False
        if lora_dropout > 0 and qbits_customop_available:
            self.lora_dropout = torch.nn.ModuleDict(
                {adapter_name: DropoutQBits(p=lora_dropout)}
            )

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            print(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        w_dequant = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=list(self.lora_A.values())[0].weight.dtype,
        )
        torch.ops.bestlaop.woq_dequantize(
            self.weight.data,
            w_dequant,
            True,
            self.compute_dtype,
            self.weight_dtype,
            self.scale_dtype,
        )
        w_data = w_dequant
        for active_adapter in self.active_adapters:
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = w_data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    w_data = orig_weights
                else:
                    w_data += self.get_delta_weight(active_adapter)

        weight = torch.ops.bestlaop.woq_quantize(
            w_data,
            True,
            self.blocksize,
            self.compute_dtype,
            self.weight_dtype,
            self.scale_dtype,
            False if self.scheme == "sym" else True,
        )

        self.weight = ParamsQBits(
            data=weight,
            requires_grad=False,
            quant_state={"scheme": self.scheme},
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_dtype=self.weight_dtype,
            scale_dtype=self.scale_dtype,
        )

    def unmerge(self) -> None:
        if not self.merged:
            print("Already unmerged. Nothing to do.")
            return

        w_dequant = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=list(self.lora_A.values())[0].weight.dtype,
        )
        torch.ops.bestlaop.woq_dequantize(
            self.weight.data,
            w_dequant,
            True,
            self.compute_dtype,
            self.weight_dtype,
            self.scale_dtype,
        )

        w_data = w_dequant
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                w_data -= self.get_delta_weight(active_adapter)

        weight = torch.ops.bestlaop.woq_quantize(
            w_data,
            True,
            self.blocksize,
            self.compute_dtype,
            self.weight_dtype,
            self.scale_dtype,
            False if self.scheme == "sym" else True,
        )

        self.weight = ParamsQBits(
            data=weight,
            requires_grad=False,
            quant_state={"scheme": self.scheme},
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_dtype=self.weight_dtype,
            scale_dtype=self.scale_dtype,
        )

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (
            transpose(
                self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                False,
            )
            * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = super().forward(x)
        elif self.merged:
            result = super().forward(x)
        else:
            result = super().forward(x)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result += output

        return result

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter


class QBitsLoraModel(LoraModel):
    _create_new_module_ = LoraModel._create_new_module

    def _create_new_module(self, lora_config, adapter_name, target, **kwargs):
        if isinstance(target, QuantizedLinearQBits):
            bias = kwargs.pop("bias", False)
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                print(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            kwargs["compute_dtype"] = target.compute_dtype
            kwargs["compress_statistics"] = target.compress_statistics
            kwargs["weight_dtype"] = target.weight_dtype
            kwargs["scale_dtype"] = target.scale_dtype
            kwargs["blocksize"] = target.blocksize
            kwargs["scheme"] = target.scheme
            new_module = QuantizedLoraLinearQBits(
                adapter_name, in_features, out_features, bias=bias, **kwargs
            )
        else:
            new_module = QBitsLoraModel._create_new_module_(
                lora_config, adapter_name, target, **kwargs
            )
        return new_module


PEFT_TYPE_TO_MODEL_MAPPING[PeftType.LORA] = QBitsLoraModel
