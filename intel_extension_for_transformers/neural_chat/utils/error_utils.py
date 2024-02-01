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

from ..errorcode import ErrorCodes
from intel_extension_for_transformers.utils import logger

_latest_error = None

def set_latest_error(error_code):
    global _latest_error
    _latest_error = error_code
    logger.error(f"neuralchat error: {ErrorCodes.error_strings[error_code]}")

def get_latest_error():
    return _latest_error

def get_latest_error_string():
    return ErrorCodes.error_strings[_latest_error] if _latest_error is not None else None

def clear_latest_error():
    global _latest_error
    _latest_error = None
