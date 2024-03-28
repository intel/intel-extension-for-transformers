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

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=E0401
import jsonlines
import re
import logging
from intel_extension_for_transformers.neural_chat.prompts.prompt import TRUTHGENERATE_PROMPT
from transformers import GenerationConfig
import argparse

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def document_set(document_file_jsonl_path):
    document_list = []
    with open(document_file_jsonl_path) as file:
        for stu in jsonlines.Reader(file):
            passages=[stu["query"],stu["pos"][0]]
            document_list.append(passages)
    return document_list

def raw_data_generate(model_id,
                      base_dir,
                      file_json_path,
                      temperature,
                      top_p,
                      top_k,
                      repetition_penalty,
                      max_new_tokens,
                      do_sample,
                      num_beams,
                      num_return_sequences,
                      use_cache):
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
   documents = document_set(base_dir)
   generation_config = GenerationConfig(
   temperature = temperature,
   top_p = top_p,
   top_k = top_k,
   repetition_penalty = repetition_penalty,
   max_new_tokens = max_new_tokens,
   do_sample = do_sample,
   num_beams = num_beams,
   num_return_sequences = num_return_sequences,
   use_cache = use_cache,
   pad_token_id=tokenizer.eos_token_id
   )

   for i in range(len(documents)):
      [question, context] = documents[i]

      if context:
         input = TRUTHGENERATE_PROMPT.format(question=question,context=context)
         if device=="cpu":
            model_input = tokenizer(input, return_tensors="pt")
         elif device=="cuda":
            model_input = tokenizer(input, return_tensors="pt").to("cuda")
         model.eval()

         with torch.no_grad():
            res = model.generate(**model_input, generation_config=generation_config)[0]
            res=tokenizer.decode(res, skip_special_tokens=True)

         res = res[res.find('Generated ground_truth:') :]
         res = re.sub('Generated ground_truth:', '', res)
         res = re.sub('---', '', res)

         result_str=res.replace('#', " ").replace(r'\t', " ").replace('\n', ' ').replace('\n\n', ' ').strip()

         if result_str and result_str.isspace()==False:
            data = {
                     "question": question,
                     "context": [context],
                     "ground_truth": result_str,
               }
            with jsonlines.open(file_json_path,"a") as file_json:
                  file_json.write(data)

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--llm_model", type=str)
   parser.add_argument("--input", type=str)
   parser.add_argument("--output", type=str)

   parser.add_argument("--temperature", type=float, default=0.8)
   parser.add_argument("--top_p", type=float, default=0.9)
   parser.add_argument("--top_k", type=int, default=40)
   parser.add_argument("--repetition_penalty", type=float, default=2.0)
   parser.add_argument("--max_new_tokens", type=int, default=48)
   parser.add_argument("--do_sample", type=bool, default=True)
   parser.add_argument("--num_beams", type=int, default=2)
   parser.add_argument("--num_return_sequences", type=int, default=2)
   parser.add_argument("--use_cache", type=bool, default=True)

   args = parser.parse_args()

   llm_model = args.llm_model
   input = args.input
   output = args.output

   temperature = args.temperature
   top_p = args.top_p
   top_k = args.top_k
   repetition_penalty = args.repetition_penalty
   max_new_tokens = args.max_new_tokens
   do_sample = args.do_sample
   num_beams = args.num_beams
   num_return_sequences = args.num_return_sequences
   use_cache = args.use_cache

   raw_data_generate(llm_model,
                     input,
                     output,
                     temperature,
                     top_p,
                     top_k,
                     repetition_penalty,
                     max_new_tokens,
                     do_sample,
                     num_beams,
                     num_return_sequences,
                     use_cache)

if __name__ == '__main__':
    main()
