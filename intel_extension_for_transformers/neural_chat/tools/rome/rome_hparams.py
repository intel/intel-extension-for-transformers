#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
import json
from dataclasses import dataclass
from typing import List


@dataclass
class ROMEHyperParams:
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    @classmethod
    def from_json(cls, fpath: os.PathLike):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_name(cls, name: str):
        data = dict(
            layers=[5],
            fact_token="subject_last",
            v_num_grad_steps=15,
            v_lr=1e-1,
            v_loss_layer=27,
            v_weight_decay=1e-3,
            clamp_norm_factor=4,
            kl_factor=0.0625,
            mom2_adjustment=True,
            rewrite_module_tmp="transformer.h.{}.mlp.fc_out",
            layer_module_tmp="transformer.h.{}",
            mlp_module_tmp="transformer.h.{}.mlp",
            attn_module_tmp="transformer.h.{}.attn",
            ln_f_module="transformer.ln_f",
            lm_head_module="lm_head",
            mom2_dataset="iohadrubin/wikitext-103-raw-v1",#"wikipedia",
            mom2_n_samples=10000,
            mom2_dtype="float32"
        )

        if name == "gpt-j-6b":
            pass
        elif name == "llama-7b":
            r"""
            Supports: LLaMA-7B, LLaMA-2-7B, Baichuan-7B, InternLM-7B...
            """
            data.update(dict(
                layers=[14],
                v_loss_layer=31,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "llama-13b":
            r"""
            Supports LLaMA-13B, LLaMA-2-13B, Baichuan-13B...
            """
            data.update(dict(
                layers=[10],
                v_loss_layer=39,
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
                mlp_module_tmp="model.layers.{}.mlp",
                attn_module_tmp="model.layers.{}.self_attn",
                ln_f_module="model.norm"
            ))
        elif name == "falcon-7b":
            data.update(dict(
                v_loss_layer=31,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        elif name == "bloom-7b1":
            data.update(dict(
                v_lr=2e-1,
                v_loss_layer=29,
                rewrite_module_tmp="transformer.h.{}.mlp.dense_4h_to_h",
                attn_module_tmp="transformer.h.{}.self_attention"
            ))
        else:
            raise NotImplementedError

        return cls(**data)
