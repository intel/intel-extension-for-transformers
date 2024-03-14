#!/usr/bin/env python
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

import numpy as np
import shutil
import torch
import unittest

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import (
    AutoModel,
    RtnConfig,
    AutoModelForCausalLM
)
from neural_speed.convert import convert_model
from neural_speed import Model

def cmpData(numa, numb):
    totalErr = ((np.abs(numa - numb))**2).sum()
    totalNum = (np.abs(numa)**2).sum()
    diff2 = np.sqrt(totalErr/totalNum)

    cos = np.dot(numa, numb)/(np.linalg.norm(numa)*np.linalg.norm(numb))
    return {"diff2": diff2, "cos": cos}

class TestLLMRUNTIME(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./runtime_outs", ignore_errors=True)

    def test_llm_runtime(self):
        model_name = "/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat/Llama-2-7b-chat-hf"
        prompt = "What is the meaning of life?"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt")

        pt_logits = torch.load("/tf_dataset2/inc-ut/nlptoolkit_ut_model/llama2_pt_logits.pth")[:,-1]
        pt_generate_ids = torch.load("/tf_dataset2/inc-ut/nlptoolkit_ut_model/llama2_pt_generate_ids.pth")[0].tolist()
        print(tokenizer.decode(pt_generate_ids))

        # check output ids
        woq_config = RtnConfig(use_quant=False)
        itrex_model = AutoModel.from_pretrained(model_name, quantization_config=woq_config, use_neural_speed=True, trust_remote_code=True)
        itrex_generate_ids = itrex_model.generate(inputs.input_ids, do_sample=False, max_new_tokens=100)[0]
        print(tokenizer.decode(itrex_generate_ids))
        for i in range(len(pt_generate_ids)):
            self.assertEqual(pt_generate_ids[i], itrex_generate_ids[i])

        # check diff of logits
        woq_configs = {
            "fp32": RtnConfig(use_quant=False),
            # "ggml_int4": RtnConfig(compute_dtype="int8", weight_dtype="int4",use_ggml=True),
            "jblas_int4": RtnConfig(bits=8, compute_dtype="int8", weight_dtype="int4"),
            # "jblas_int8": RtnConfig(compute_dtype="bf16", weight_dtype="int8"),
            }
        for config_type in woq_configs:
            itrex_model = AutoModel.from_pretrained(model_name, quantization_config=woq_configs[config_type],
                                                    use_neural_speed=True, trust_remote_code=True)
            itrex_logits = itrex_model(inputs.input_ids)
            print(config_type, cmpData(pt_logits.detach().numpy().flatten(), itrex_logits.flatten()))


    def test_gguf_api(self):
        model_name = "TheBloke/Mistral-7B-v0.1-GGUF"
        model_file = "mistral-7b-v0.1.Q4_0.gguf"
        tokenizer_name = "/tf_dataset2/models/pytorch/Mistral-7B-v0.1"

        prompt = "Once upon a time"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        streamer = TextStreamer(tokenizer)

        model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
        output = model.generate(inputs, streamer=streamer, max_new_tokens=10)
        print("output = ", output)
        assert(output == [[1, 5713, 3714, 264, 727, 28725, 736, 403, 264, 1628, 2746, 693, 6045, 298, 1220, 28723, 985]])


    def test_beam_search(self):
        model_name = "/tf_dataset2/models/pytorch/gpt-j-6B"  # or local path to model
        prompts = [
           "she opened the door and see",
           "tell me 10 things about jazz music",
           "What is the meaning of life?",
           "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer"\
            " The slings and arrows of outrageous fortune, "\
            "Or to take arms against a sea of troubles."\
            "And by opposing end them. To die—to sleep,"
            ]

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                                  padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]
        inputs = tokenizer(prompts, padding=True, return_tensors='pt')

        # pytorch fp32
        pt_generate_ids = torch.load("/tf_dataset2/inc-ut/nlptoolkit_ut_model/beam_pt_generate_ids.pth").tolist()

        # llm runtime fp32
        woq_config = RtnConfig(use_quant=False)
        itrex_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=woq_config, trust_remote_code=True)
        itrex_generate_ids = itrex_model.generate(
            inputs.input_ids, num_beams=4, max_new_tokens=128, min_new_tokens=30, early_stopping=True, pad_token=pad_token)
        for i in range(len(itrex_generate_ids)):
            self.assertListEqual(pt_generate_ids[i], itrex_generate_ids[i])


if __name__ == "__main__":
    unittest.main()
