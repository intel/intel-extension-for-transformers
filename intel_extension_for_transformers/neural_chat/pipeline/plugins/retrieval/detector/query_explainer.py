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
"""Polish and clarify the input user query with LLM."""

from intel_extension_for_transformers.neural_chat.pipeline.plugins.prompt.prompt_template \
     import polish_query_prompt
from intel_extension_for_transformers.neural_chat.models.model_utils import predict


class QueryPolisher:
    def __init__(self):
        pass

    def polish_query(self, model_name, query):
        """Using the LLM to polish the user query to fix the gramma errors and clarify the user query."""
        prompt = polish_query_prompt(query)
        params = {}
        params["model_name"] = model_name
        params["prompt"] = prompt
        params["temperature"] = 0.1
        params["top_k"] = 3
        params["max_new_tokens"] = 512
        params['do_sample'] = True
        new_query = predict(**params)
        return new_query
