"""init."""
# !/usr/bin/env python
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

from .pruning import Pruning
from packaging.version import Version
from neural_compressor import __version__
from neural_compressor.config import WeightPruningConfig
# pylint: disable=E0611
if Version(__version__).release > Version('2.1.1').release:  # pragma: no cover
    from neural_compressor.compression.pruner.model_slim import model_slim, parse_auto_slim_config