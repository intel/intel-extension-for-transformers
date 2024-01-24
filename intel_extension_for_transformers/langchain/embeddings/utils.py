# !/usr/bin/env python
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

import os
from typing import Union, Optional
from intel_extension_for_transformers.transformers.utils.utility import LazyImport
sentence_transformers = LazyImport("sentence_transformers")

def get_module_path(model_name_or_path: str, 
                    path: str,
                    token: Optional[Union[bool, str]], 
                    cache_folder: Optional[str]):
    is_local = os.path.isdir(model_name_or_path)
    if is_local:
        return os.path.join(model_name_or_path, path)
    else:
        return sentence_transformers.util.load_dir_path(
            model_name_or_path, path, token=token, cache_folder=cache_folder)
