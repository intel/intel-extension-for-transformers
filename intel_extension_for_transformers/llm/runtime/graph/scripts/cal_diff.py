#!/usr/bin/env python
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

import numpy as np
import argparse
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModel, WeightOnlyQuantConfig, AutoModelForCausalLM


def cmpData(numa, numb):
    totalErr = ((numa - numb)**2).sum()
    totalNum = (numa**2).sum()
    diff2 = np.sqrt(totalErr/totalNum)

    cos = np.dot(numa, numb)/(np.linalg.norm(numa)*np.linalg.norm(numb))
    return {"diff2": diff2, "cos": cos}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diff for a model")
    parser.add_argument('--model_name', type=str, default="~/Llama-2-7b-chat-hf")
    args = parser.parse_args()

    woq_configs = {
        "fp32": WeightOnlyQuantConfig(use_gptq=True),
        # "ggml_int4": WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_cache=True, use_ggml=True),
        # "jblas_int4": WeightOnlyQuantConfig(compute_dtype="int8", weight_dtype="int4", use_cache=True),
        # "jblas_int8": WeightOnlyQuantConfig(compute_dtype="bf16", weight_dtype="int8", use_cache=True),
        }
    prompt = "What is the meaning of life?"

    model_name = "/mnt/disk1/data2/zhenweil/models/mistral/neural-chat-7b-v3-1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    pt_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    pt_model.eval() 
    pt_logits = pt_model(input_ids=inputs.input_ids).logits[:,-1]

    for config_type in woq_configs:
        itrex_model = AutoModel.from_pretrained("/mnt/disk1/data2/zhenweil/models/mistral/neural-chat-7b-v3-1-GPTQ", quantization_config=woq_configs[config_type], 
                                                use_llm_runtime=True, trust_remote_code=True)
        itrex_logits = itrex_model(inputs.input_ids)

        print(config_type, cmpData(pt_logits.detach().numpy().flatten(), itrex_logits.flatten()))
