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

try:
    import habana_frameworks.torch.hpu as hthpu
    is_hpu_available = True
except ImportError:
    is_hpu_available = False

try:
    import intel_extension_for_pytorch as intel_ipex
    is_ipex_available = True
except ImportError:
    is_ipex_available = False

def get_device_type():
    if torch.cuda.is_available():
        device = "cuda"
    elif is_hpu_available:
        device = "hpu"
    elif is_ipex_available and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    return device

def is_audio_file(filename):
    audio_extensions = ['mp3', 'wav', 'flac', 'ogg', 'aac', 'm4a']
    file_extension = filename.split('.')[-1].lower()

    if file_extension in audio_extensions:
        return True
    else:
        return False

def is_openai_model(model_name_or_path):
    # Check https://platform.openai.com/docs/models/model-endpoint-compatibility
    return any(name in model_name_or_path for name in ["gpt-4", "gpt-3.5-turbo"])

def is_hf_model(model_name_or_path):
    return "http" in model_name_or_path

def get_device_type():
    if torch.cuda.is_available():
        device = "cuda"
    elif is_hpu_available:
        device = "hpu"
    elif is_ipex_available and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    return device

def supported_gpus():
    return ['flex', 'max', 'arc']

def get_gpu_family():
    ''' Get gpu device family info.

    Return 'flex'|'max'|'arc'| 'no_gpu'| assert

    Note, this function need to import intel_extension_for_pytorch


    Additional info (common gpu name):
      'Intel(R) Data Center GPU Flex 170'
      'Intel(R) Data Center GPU Max 1100'
      'Intel(R) Arc(TM) A770 Graphics'
    '''

    import intel_extension_for_pytorch as ipex
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return 'no_gpu'

    name = torch.xpu.get_device_name()
    if 'GPU Flex' in name:
        result = 'flex'
    elif 'GPU Max' in name:
        result = 'max'
    elif 'Arc(TM)' in name:
        result = 'arc'
    else:
        assert False, "Unsupported GPU device: {}".format(name)

    if result not in supported_gpus():
        assert False, "Unsupported GPU device: {}".format(name)
    else:
        return result

_autoround_available = importlib.util.find_spec("auto_round") is not None
_autoround_version = "N/A"
if _autoround_available:
    try:
        _autoround_version = importlib_metadata.version("auto_round")
    except importlib_metadata.PackageNotFoundError:
        _autoround_available = False

def is_autoround_available():
    return _autoround_available
