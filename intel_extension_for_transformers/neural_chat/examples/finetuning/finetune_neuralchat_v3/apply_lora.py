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

import argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def apply_lora(base_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    return base, model, base_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--lora-model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()

    base, target, base_tokenizer = apply_lora(args.base_model_path, args.lora_model_path)
    target.save_pretrained(args.output_path)
    base_tokenizer.save_pretrained(args.output_path)
