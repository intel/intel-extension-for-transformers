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

from ..constants import ErrorCodes

class ErrorManager:
    _latest_error = None

    @classmethod
    def set_latest_error(cls, error_code):
        cls._latest_error = error_code

    @classmethod
    def get_latest_error(cls):
        return cls._latest_error

    @classmethod
    def get_latest_error_string(cls):
        return ErrorCodes.error_strings[cls._latest_error]
