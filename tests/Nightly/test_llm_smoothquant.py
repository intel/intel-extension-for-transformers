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

from math import isclose
from intel_extension_for_transformers.transformers import (
    AutoModelForCausalLM,
    SmoothQuantConfig,
)
from transformers import AutoTokenizer
from packaging.version import Version
import neural_compressor.adaptor.pytorch as nc_torch
import unittest
import torch

PT_VERSION = nc_torch.get_torch_version()


@unittest.skipIf(
    PT_VERSION.release < Version("2.1.0").release,
    "Please use PyTroch 2.1.0 or higher version for executor backend",
)
class TestLLMQuantization(unittest.TestCase):
    def test_qwen(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-7B-Chat", trust_remote_code=True
        )
        sq_config = SmoothQuantConfig(
            calib_iters=3,
            calib_len=5,
            tokenizer=tokenizer,
            excluded_precisions=["bf16"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-7B-Chat",
            quantization_config=sq_config,
            trust_remote_code=True,
            use_neural_speed=False,
        )
        self.assertTrue(isinstance(model.model, torch.jit.ScriptModule))

    def test_chatglm2(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b", trust_remote_code=True
        )
        sq_config = SmoothQuantConfig(calib_iters=3, calib_len=5, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm2-6b",
            quantization_config=sq_config,
            trust_remote_code=True,
            use_neural_speed=False,
        )
        self.assertTrue(isinstance(model.model, torch.jit.ScriptModule))

    def test_chatglm3(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm3-6b", trust_remote_code=True
        )
        sq_config = SmoothQuantConfig(calib_iters=3, calib_len=5, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm3-6b",
            quantization_config=sq_config,
            trust_remote_code=True,
            use_neural_speed=False,
        )
        self.assertTrue(isinstance(model.model, torch.jit.ScriptModule))


if __name__ == "__main__":
    unittest.main()
