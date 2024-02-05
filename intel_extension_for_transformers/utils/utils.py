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

"""Utility."""

import importlib
import sys
import torch
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


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

_ipex_available = importlib.util.find_spec("intel_extension_for_pytorch") is not None
_ipex_version = "N/A"
if _ipex_available:
    try:
        _ipex_version = importlib_metadata.version("intel_extension_for_pytorch")
    except importlib_metadata.PackageNotFoundError:
        _ipex_available = False

def is_ipex_available():
    return _ipex_available
