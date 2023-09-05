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

import os
import torch
import transformers
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval import Retriever
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.detector import IntentDetector
from intel_extension_for_transformers.neural_chat.pipeline.plugins.retrieval.indexing import DocumentIndexing
from intel_extension_for_transformers.neural_chat.pipeline.plugins.prompt import generate_qa_prompt, generate_prompt
from intel_extension_for_transformers.neural_chat.plugins import register_plugin


@register_plugin("retrieval")
class Agent_QA():
    def __init__(self, persist_dir="./output", process=True, input_path=None,
                 embedding_model="hkunlp/instructor-large", max_length=512, retrieval_type="dense",
                 document_store=None, top_k=1, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}):
        self.model = None
        self.tokenizer = None
        self.retrieval_type = retrieval_type

        self.intent_detector = IntentDetector()
        if os.path.exists(input_path):
            self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type, document_store=document_store,
                                               persist_dir=persist_dir, process=process,
                                               embedding_model=embedding_model, max_length=max_length)
            self.db = self.doc_parser.KB_construct(input_path)
            self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=self.db, top_k=top_k,
                                       search_type=search_type, search_kwargs=search_kwargs)



    def pre_llm_inference_actions(self, model_name, query):
        intent = self.intent_detector.intent_detection(model_name, query)

        if 'qa' not in intent.lower():
            print("Chat with AI Agent.")
            prompt = generate_prompt(query)
        else:
            print("Chat with QA agent.")
            context = self.retriever.get_context(query)
            prompt = generate_qa_prompt(query, context)
        return prompt

