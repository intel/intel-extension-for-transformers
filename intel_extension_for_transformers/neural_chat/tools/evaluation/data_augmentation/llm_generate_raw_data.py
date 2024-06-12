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

import os
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer  # pylint: disable=E0401
import jsonlines
import os, re
from typing import List
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.parser.parser import DocumentParser
import logging
from intel_extension_for_transformers.neural_chat.prompts.prompt import QUERYGENERATE_PROMPT
from transformers import GenerationConfig

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def document_append(data_collection):
    documents = []
    for data, metadata in data_collection:
        if len(data) < 5:
            continue
        documents.append(data)
    return documents

def raw_data_generate(model_id,
                      input_path,
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
   data_collection = DocumentParser().load(input=input_path)
   documents = document_append(data_collection)

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
      context = documents[i]

      if context:
         input = QUERYGENERATE_PROMPT.format(context=context)
         if device=="cpu":
            model_input = tokenizer(input, return_tensors="pt")
         elif device=="cuda":
            model_input = tokenizer(input, return_tensors="pt").to("cuda")
         model.eval()
         result = []

         for j in range(5):
            with torch.no_grad():
               res = model.generate(**model_input, generation_config=generation_config)[0]
               res=tokenizer.decode(res, skip_special_tokens=True)

            res = res[res.find('Generated questions:') :]
            res = re.sub('Generated questions:', '', res)
            res = re.sub('---', '', res)

            res = res.split("?")[0:2]
            for r in res:
               r = r.replace('1.', "").replace('2.', "")
               r = r.replace('Evaluation:', "")
               r = r.replace('#', " ").replace(r'\t', " ").replace('\n', ' ').replace('\n\n', ' ').strip()
               r = r + '?'
               result.append(r)

         result_str=''
         result_set = list(set(result))
         for k in range(len(result_set)):
            result_str = result_str + str(k) + '. '+ result_set[k]

         if result_str and result_str.isspace()==False:
            data = {
                     "query": result_str,
                     "pos": [context],
               }
            with jsonlines.open(file_json_path,"a") as file_json:
                  file_json.write(data)
