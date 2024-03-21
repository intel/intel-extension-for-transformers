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

import importlib
import sys
import torch
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

try:
    import habana_frameworks.torch.hpu as hthpu
    is_hpu_available = True
except ImportError:
    is_hpu_available = False

_ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
_ipex_version = "N/A"
if _ipex_available:
    try:
        _ipex_version = importlib_metadata.version("intel_extension_for_pytorch")
    except importlib_metadata.PackageNotFoundError:
        _ipex_available = False

def is_ipex_available():
    return _ipex_available

def get_device_type():
    if torch.cuda.is_available():
        device = "cuda"
    elif is_hpu_available:
        device = "hpu"
    elif is_ipex_available() and torch.xpu.is_available():
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

def supported_gpus():
    return ['flex', 'max', 'arc']
