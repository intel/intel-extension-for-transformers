#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

"""Utils for pytorch framework."""

import importlib
import os
from neural_compressor.utils.utility import LazyImport


CONFIG_NAME = "best_configure.yaml"
ENGINE_MODEL_NAME = "model.bin"
ENGINE_MODEL_CONFIG = "conf.yaml"
ENCODER_NAME = "encoder_model.bin"
DECODER_NAME = "decoder_model.bin"
DECODER_WITH_PAST_NAME = "decoder_with_past_model.bin"
WEIGHTS_NAME = "pytorch_model.bin"


def distributed_init(backend="gloo", world_size=1, rank=-1, init_method=None,
                     master_addr='127.0.0.1', master_port='12345'):
    """Init the distibute environment."""
    torch = LazyImport("torch")
    rank = int(os.environ.get("RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", master_addr)
        master_port = os.environ.get("MASTER_PORT", master_port)
        init_method = 'env://{addr}:{port}'.format(addr=master_addr, port=master_port)
    torch.distributed.init_process_group(
        backend,
        init_method=init_method,
        world_size=world_size, 
        rank=rank
    )


def remove_label(input):
    if "labels" in input:  # for GLUE
        input.pop('labels')
    elif "start_positions" in input and "end_positions" in input:  # for SQuAD
        # pragma: no cover
        input.pop('start_positions')
        input.pop('end_positions')
    return input
