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

from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModel, RtnConfig

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
        itrex_model = AutoModel.from_pretrained(model_name, load_in_4bit=True,
                                                use_neural_speed=True, trust_remote_code=True)
        itrex_logits = itrex_model(inputs.input_ids)
        cmp = cmpData(pt_logits.detach().numpy().flatten(), itrex_logits.flatten())
        print("load_in_4bit: ", cmp)
        self.assertTrue(cmp["diff2"] < 0.42)

        itrex_model = AutoModel.from_pretrained(model_name, load_in_8bit=True,
                                                use_neural_speed=True, trust_remote_code=True)
        itrex_logits = itrex_model(inputs.input_ids)
        cmp = cmpData(pt_logits.detach().numpy().flatten(), itrex_logits.flatten())
        print("load_in_8bit: ", cmp)
        self.assertTrue(cmp["diff2"] < 0.42)

if __name__ == "__main__":
    unittest.main()
