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

from .llm_generate_raw_data import raw_data_generate
from .mine_hard_negatives_check_similarity import mine_hard_negatives, similarity_check
import argparse
import os

def construct_retrieval_dataset(
      llm_model,
      embedding_model,
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
      use_cache,
      range_for_sampling,
      negative_number,
      use_gpu_for_searching,
      similarity_threshold):

   output_path=output+'/raw.jsonl'
   raw_data_generate(llm_model,
                     input,
                     output_path,
                     temperature,
                     top_p,
                     top_k,
                     repetition_penalty,
                     max_new_tokens,
                     do_sample,
                     num_beams,
                     num_return_sequences,
                     use_cache)

   output_hn_path=output+'/minedHN.jsonl'
   mine_hard_negatives(embedding_model,
                       output_path,
                       output_hn_path,
                       range_for_sampling,
                       negative_number,
                       use_gpu_for_searching)

   output_json_split_path = output+"/minedHN_split.jsonl"
   similarity_check(output_hn_path,
                    output_json_split_path,
                    embedding_model,
                    similarity_threshold)


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--llm_model", type=str)
   parser.add_argument("--embedding_model", type=str)
   parser.add_argument("--input", type=str)
   parser.add_argument("--output", type=str, default='./data')

   parser.add_argument("--temperature", type=float, default=0.8)
   parser.add_argument("--top_p", type=float, default=0.9)
   parser.add_argument("--top_k", type=int, default=40)
   parser.add_argument("--repetition_penalty", type=float, default=2.0)
   parser.add_argument("--max_new_tokens", type=int, default=48)
   parser.add_argument("--do_sample", type=bool, default=True)
   parser.add_argument("--num_beams", type=int, default=2)
   parser.add_argument("--num_return_sequences", type=int, default=2)
   parser.add_argument("--use_cache", type=bool, default=True)

   parser.add_argument("--range_for_sampling", type=str, default='2-10')
   parser.add_argument("--negative_number", type=int, default=5)
   parser.add_argument("--use_gpu_for_searching", type=bool, default=False)

   parser.add_argument("--similarity_threshold", type=float, default=0.6)

   args = parser.parse_args()

   llm_model = args.llm_model
   embedding_model = args.embedding_model
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

   range_for_sampling=args.range_for_sampling
   negative_number=args.negative_number
   use_gpu_for_searching=args.use_gpu_for_searching

   similarity_threshold=args.similarity_threshold

   try:
      if os.path.exists(output) == False:
         os.mkdir(output)
      else:
         if os.path.exists(output+'/raw.jsonl'):
            os.remove(output+'/raw.jsonl')
         if os.path.exists(output+'/minedHN.jsonl'):
            os.remove(output+'/minedHN.jsonl')
         if os.path.exists(output+'/minedHN_split.jsonl'):
            os.remove(output+'/minedHN_split.jsonl')
   except:
      pass
   
   construct_retrieval_dataset(
      llm_model,
      embedding_model,
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
      use_cache,
      range_for_sampling,
      negative_number,
      use_gpu_for_searching,
      similarity_threshold)

if __name__ == '__main__':
    main()
