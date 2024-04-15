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

from transformers.generation import GenerationConfig


class GaudiGenerationConfig(GenerationConfig):
    """
    This class extends [`transformers.generation.GenerationConfig`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py)
    to add HPU-specific arguments for generation.

    Arg:
    trim_logit (`bool`, *optional):
        Calculate logits only for the last token to save memory in the first step.
    static_shapes (`bool`, *optional*):
        Whether to use static shapes for generation or not. It will run faster on HPUs with static shapes
        but not all models support it. If not specified, it will automatically be set to `True` if the given
        model supports it.
    ignore_eos (`bool`, *optional*):
        Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
        If not specified, it will automatically be set to `True` if lazy mode is on.
    attn_softmax_bf16 (`bool`, *optional*):
        Whether to run attention softmax layer in lower precision provided that the model supports it and
        is also running in lower precision.
    limit_hpu_graphs (`bool`, *optional*):
        Skip HPU Graph usage for first token to save memory
    reuse_cache (`bool`, *optional*):
        Whether to reuse key/value cache for decoding. It should save memory.
    bucket_size (`int`, *optional*):
        If negative (default=-1) pad to max if `static_shapes` is set. Else start with
        `shape = bucket_size * ceil(prompt_len/bucket_size)` and then grow space by `bucket_size` when needed.
        Only active if `static_shapes` is used. Can't be used with `reuse_cache`.
    bucket_internal (`bool`, *optional*):
        Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.
    use_flash_attention (`bool`, *optional*):
        Whether to use flash attention optimization.
    flash_attention_recompute (`bool`, *optional*):
        Whether to enable recompute if use Habana flash attention.
    flash_attention_causal_mask (`bool`, *optional*):
        Whether to enable causal_mask if use Habana flash attention.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trim_logits = kwargs.get("trim_logits", None)
        self.static_shapes = kwargs.get("static_shapes", None)
        self.ignore_eos = kwargs.get("ignore_eos", None)
        self.attn_softmax_bf16 = kwargs.get("attn_softmax_bf16", None)
        self.limit_hpu_graphs = kwargs.get("limit_hpu_graphs", None)
        self.reuse_cache = kwargs.get("reuse_cache", None)
        self.bucket_size = kwargs.get("bucket_size", -1)
        self.bucket_internal = kwargs.get("bucket_internal", None)
        self.reduce_recompile = kwargs.get("reduce_recompile", None)
        self.use_flash_attention = kwargs.get("use_flash_attention", None)
        self.flash_attention_recompute = kwargs.get("flash_attention_recompute", None)
        self.flash_attention_causal_mask = kwargs.get("flash_attention_causal_mask", None)
        self.use_fused_rope = kwargs.get("use_fused_rope", None)
