# !/usr/bin/env python
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

# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from https://github.com/mit-han-lab/streaming-llm/tree/main
"""


from .attention_sink_kv_cache import AttentionSinkKVCache


def enable_streaming_llm(model, attention_sink_size, window_size):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from .models.llama.pos_shift_llama import (
            enable_gaudi_llama_pos_shift_attention,
            enable_gaudi_llama_cont_cat_kv_cache
        )

        enable_gaudi_llama_pos_shift_attention(model)
        enable_gaudi_llama_cont_cat_kv_cache(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = AttentionSinkKVCache(
        attention_sink_size=attention_sink_size,
        attention_sink_window_size=window_size - attention_sink_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache
