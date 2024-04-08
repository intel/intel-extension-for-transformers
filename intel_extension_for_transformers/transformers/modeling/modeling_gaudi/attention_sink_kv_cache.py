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
Adapted from https://github.com/tomaarsen/attention_sinks
"""

from dataclasses import dataclass
import torch

def slice1d(x, start, end):
    return x[:, start:end, ...]


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

@dataclass
class AttentionSinkKVCache:
    attention_sink_size: int = 4
    attention_sink_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2
    kv_dtype: str = "fp32"

    def __post_init__(self):
        self.cache_size = self.attention_sink_size + self.attention_sink_window_size
        self.k_slice = DIM_TO_SLICE[self.k_seq_dim]
        self.v_slice = DIM_TO_SLICE[self.v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.attention_sink_size),
                        self.k_slice(k, seq_len - self.attention_sink_window_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.attention_sink_size),
                        self.v_slice(v, seq_len - self.attention_sink_window_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.attention_sink_size),
                        self.k_slice(
                            k,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.attention_sink_size),
                        self.v_slice(
                            v,
                            seq_len - self.attention_sink_window_size + num_coming,
                            seq_len,
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
