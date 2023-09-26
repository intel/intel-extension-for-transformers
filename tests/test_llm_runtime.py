import numpy
import shutil
import torch
import unittest

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig


class TestLLMRUNTIME(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./ne_chatglm_q.bin", ignore_errors=True)

    def test_llm_runtime(self):

        model_name = "/tf_dataset2/models/pytorch/chatglm2-6b"  # or local path to model
        woq_config = WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4")
        prompt = "小明的妈妈有三个孩子，老大叫大毛，老二叫二毛，老三叫什么？"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        streamer = TextStreamer(tokenizer)

        model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_llm_runtime=True, trust_remote_code=True)
        gen_tokens = model.generate(input_ids, streamer=streamer, max_new_tokens=300)