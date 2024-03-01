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
from modelscope import AutoModelForCausalLM, AutoTokenizer
import jsonlines
import os, re
from typing import List
from context_utils import load_unstructured_data, load_structured_data, get_chuck_data
from html_parser import load_html_data
import logging
from intel_extension_for_transformers.neural_chat.prompts.prompt import QUERYGENERATE_PROMPT
from transformers import GenerationConfig

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

class DocumentLoading:
    def __init__(self, max_chuck_size=512, min_chuck_size=5, process=True):
        """
        Wrapper for document parsing.
        """
        self.max_chuck_size = max_chuck_size
        self.min_chuck_size = min_chuck_size
        self.process = process


    def data_load(self, input, **kwargs):
        """
        The API for loading the file. Support single file, batch files, and urls parsing.
        """
        if 'max_chuck_size' in kwargs:
            self.max_chuck_size=kwargs['max_chuck_size']
        if 'min_chuck_size' in kwargs:
            self.min_chuck_size = kwargs['min_chuck_size']
        if 'process' in kwargs:
            self.process = kwargs['process']

        if isinstance(input, str):
            if os.path.isfile(input):
                data_collection = self.parse_document(input)
            elif os.path.isdir(input):
                data_collection = self.batch_parse_document(input)
            else:
                logging.error("Please check your upload file and try again!")
        elif isinstance(input, List):
            try:
                data_collection = self.parse_html(input)
            except:
                logging.error("The given link/str is unavailable. Please try another one!")
        else:
            logging.error("The input format is invalid!")

        documents = []
        for data, metadata in data_collection:
             if len(data) < 5:
                continue
             documents.append(data)
        return documents


    def parse_document(self, input):
        """
        Parse the uploaded file.
        """
        if input.endswith("pdf") or input.endswith("docx") or input.endswith("html") \
           or input.endswith("txt") or input.endswith("md"):
            content = load_unstructured_data(input)
            if self.process:
                chuck = get_chuck_data(content, self.max_chuck_size, self.min_chuck_size, input)
            else:
                chuck = [[content.strip(),input]]
        elif input.endswith("jsonl") or input.endswith("xlsx") or input.endswith("csv") or \
                input.endswith("json"):
            chuck = load_structured_data(input, self.process, \
                                         self.max_chuck_size, self.min_chuck_size)
        else:
            logging.info("This file {} is ignored. Will support this file format soon.".format(input))
            raise Exception("[Rereieval ERROR] Document format not supported!")
        return chuck

    def parse_html(self, input):
        """
        Parse the uploaded file.
        """
        chucks = []
        for link in input:
            if re.match(r'^https?:/{2}\w.+$', link):
                content = load_html_data(link)
                if content == None:
                    continue
                if self.process:
                    chuck = get_chuck_data(content, self.max_chuck_size, self.min_chuck_size, link)
                else:
                    chuck = [[content.strip(), link]]
                chucks += chuck
            else:
                logging.error("The given link/str {} cannot be parsed.".format(link))

        return chucks


    def batch_parse_document(self, input):
        """
        Parse the uploaded batch files in the input folder.
        """
        paragraphs = []
        for dirpath, dirnames, filenames in os.walk(input):
            for filename in filenames:
                if filename.endswith("pdf") or filename.endswith("docx") or filename.endswith("html") \
                    or filename.endswith("txt") or filename.endswith("md"):
                    content = load_unstructured_data(os.path.join(dirpath, filename))
                    if self.process:
                        chuck = get_chuck_data(content, self.max_chuck_size, self.min_chuck_size, input)
                    else:
                        chuck = [[content.strip(),input]]
                    paragraphs += chuck
                elif filename.endswith("jsonl") or filename.endswith("xlsx") or filename.endswith("csv") or \
                        filename.endswith("json"):
                    chuck = load_structured_data(os.path.join(dirpath, filename), \
                                                 self.process, self.max_chuck_size, self.min_chuck_size)
                    paragraphs += chuck
                else:
                    logging.info("This file {} is ignored. Will support this file format soon.".format(filename))
                    raise Exception("[Rereieval ERROR] Document format not supported!")
        return paragraphs

def raw_data_generate(model_id, base_dir, file_json_path,temperature,top_p,top_k,repetition_penalty,max_new_tokens,do_sample,num_beams,num_return_sequences,use_cache):
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)
   documents = DocumentLoading().data_load(input=base_dir)

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
