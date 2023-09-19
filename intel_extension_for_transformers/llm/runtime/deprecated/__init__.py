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
"""The module of Neural Engine."""

import subprocess
import sys
import intel_extension_for_transformers
import os.path as path


def neural_engine_bin():
    ''' Entry point for C++ interface '''
    neural_engine_bin = path.join(
        intel_extension_for_transformers.__path__[0], 'neural_engine_bin')
    raise SystemExit(subprocess.call(
        [neural_engine_bin] + sys.argv[1:], close_fds=False))
