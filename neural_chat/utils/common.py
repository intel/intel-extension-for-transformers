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

import torch
import habana_frameworks.torch.hpu as hthpu

def get_device_type():
    if torch.cuda.is_available():
        device = "cuda"
    elif hthpu.is_available():
        device = "hpu"
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    return device

def get_backend_type():
    return "ipex"