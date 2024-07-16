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

import sys
import unittest

from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers.kv_cache_compression import (
    H2OKVPruner,
    H2OConfig,
    LlamaForCausalLM
)

MODLE_NAME = "Maykeye/TinyLLama-v0"

class TestH2O(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_h2o(self):
        h2o_config = H2OConfig(
            heavy_ratio=0.1,
            recent_ratio=0.1,
            h2o_min_seqlen=-1,
            real_drop=True,
            mean=False
        )
        model = LlamaForCausalLM.from_pretrained(MODLE_NAME, prune_config=h2o_config)
        tokenizer = AutoTokenizer.from_pretrained(MODLE_NAME)
        prompt_text = "In a small, bustling cafe nestled in the heart of a vibrant city,"
        input = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
        generate_ids = model.generate(input, max_new_tokens=20)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        h2o_config = H2OConfig(
            heavy_ratio=0.1,
            recent_ratio=0.1,
            h2o_min_seqlen=-1,
            real_drop=False,
            mean=True
        )
        model = LlamaForCausalLM.from_pretrained(MODLE_NAME, prune_config=h2o_config)
        output = model(input)

if __name__ == "__main__":
    unittest.main()