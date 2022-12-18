#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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


# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

"""Utils for drop and restore function."""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


def sample_length_configuration(
    max_seq_length,
    num_hidden_layers,
    layer_config=None,
    length_drop_prob=None,
    length_drop_ratio=None,
    length_drop_ratio_bound=None,
    min_length=2,
):
    """Get different sequence length for hidden layers.
    
    Args:
        max_seq_length: A number to set the max sequence length.
        num_hidden_layers: A number of total hidden layers.
        layer_config: Config to specify layers which use the max_seq_length.
        length_drop_prob: Probability to truncate the sequence.
        length_drop_ratio: Ratio to truncate the sequence.
        length_drop_ratio_bound: The max ratio to truncate the sequence.
            If the ratio set, the length will not less than max_seq_length * ratio.
        min_length: The number to set the min sequence length.
    
    Return:
        (Tuple): The tuple of length configuration for different hidden layers.
    """
    length = max_seq_length
    length_configuration = ()
    for i in range(num_hidden_layers):
        if layer_config is None or i in layer_config:
            if length_drop_prob is not None:
                length = length - np.random.binomial(length, length_drop_prob)
            elif length_drop_ratio is not None:
                length = int(np.ceil(length * (1 - length_drop_ratio)))
            elif length_drop_ratio_bound is not None:
                length = np.random.randint(int(np.ceil(length * (1 - length_drop_ratio_bound))), length + 1)
        length = max(length, min_length)
        length_configuration += (length,)
    return length_configuration


def sample_layer_configuration(
    num_hidden_layers,
    layer_dropout_prob=None,
    layer_dropout=None,
    layer_dropout_bound=None,
):
    """Get sample layers depends on the set paramaters.
    
    Args:
        num_hidden_layers: A number to set the max sequence length.
        layer_dropout_prob: Probability to dropout a layer.
        layer_dropout: Number of how many layers to dropout.
        layer_dropout_bound: The bound of how many layers to dropout.
    
    Return:
        (Tuple): The tuple to the numbers of which samples are kept.
    """
    if layer_dropout_prob is not None:
        return tuple(i for i in range(num_hidden_layers) if np.random.random() >= layer_dropout_prob)
    elif layer_dropout is not None:
        layer_dropout = min(layer_dropout, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - layer_dropout))
    elif layer_dropout_bound is not None:
        layer_dropout_bound = min(layer_dropout_bound, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - np.random.randint(0, layer_dropout_bound + 1)))
    return None
