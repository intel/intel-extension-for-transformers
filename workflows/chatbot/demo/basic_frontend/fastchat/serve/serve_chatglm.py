# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import List, Tuple


@torch.inference_mode()
def chatglm_generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    """Generate text using model's chat api"""
    messages = params["prompt"]
    max_new_tokens = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 0.7))

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": None,
    }

    hist = []
    for i in range(0, len(messages) - 2, 2):
        hist.append((messages[i][1], messages[i + 1][1]))
    query = messages[-2][1]

    for response, new_hist in model.stream_chat(tokenizer, query, hist):
        output = query + " " + response
        yield output
