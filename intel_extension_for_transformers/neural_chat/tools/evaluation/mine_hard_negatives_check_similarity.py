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

import jsonlines
from hn_mine import find_knn_neg
from sentence_transformers import SentenceTransformer

def mine_hard_negatives(model_name_or_path, input_file, output_file, range_for_sampling, negative_number, use_gpu_for_searching):   
   candidate_pool=None

   sample_range = range_for_sampling.split('-')
   sample_range = [int(x) for x in sample_range]

   model = SentenceTransformer(model_name_or_path)

   find_knn_neg(model,
               input_file=input_file,
               candidate_pool=candidate_pool,
               output_file=output_file,
               sample_range=sample_range,
               negative_number=negative_number,
               use_gpu=use_gpu_for_searching)

def similarity_score(queries,passages,model_name_or_path):
   queries = [queries]
   passages = passages
   instruction = ""
   model = SentenceTransformer(model_name_or_path)
   q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
   p_embeddings = model.encode(passages, normalize_embeddings=True)
   similarity_score =  q_embeddings @ p_embeddings.T
   return similarity_score

def similarity_check(file_jsonl_path,file_json_split_path,model_name_or_path, similarity_threshold):
   with open(file_jsonl_path) as file:
      for stu in jsonlines.Reader(file):
         stu["query"]=stu["query"].split("?")[:-1]
         for i in range(len(stu["query"])):
               stu["query"][i]=stu["query"][i].lstrip('0123456789-. ')+ '?'
               if similarity_score(stu["query"][i],stu["pos"],model_name_or_path) >= similarity_threshold:
                  data = {
                        "query": stu["query"][i],
                        "pos": stu["pos"],
                        "neg": stu["neg"],
                     }
                  with jsonlines.open(file_json_split_path,"a") as file_json:
                     file_json.write(data)

