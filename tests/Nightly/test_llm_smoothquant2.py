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
# request transformers version lower or equal than 4.33.0
class TestLLMQuantization(unittest.TestCase):
    def test_baichuan(self):
        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base")
        sq_config = SmoothQuantConfig(calib_iters=5, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            "baichuan-inc/Baichuan2-7B-Base",
            quantization_config=sq_config,
            trust_remote_code=True,
            use_llm_runtime=False,
        )
        self.assertTrue(isinstance(model.model, torch.jit.ScriptModule))

    def test_chatglm(self):
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b")
        sq_config = SmoothQuantConfig(calib_iters=3, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm-6b",
            quantization_config=sq_config,
            trust_remote_code=True,
            use_llm_runtime=False,
        )
        self.assertTrue(isinstance(model.model, torch.jit.ScriptModule))


if __name__ == "__main__":
    unittest.main()
