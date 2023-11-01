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
from .retrieval_base import Retriever
from .detector.intent_detection import IntentDetector
from .indexing.indexing import DocumentIndexing
from intel_extension_for_transformers.neural_chat.pipeline.plugins.prompt.prompt_template \
    import generate_qa_prompt, generate_prompt, generate_qa_enterprise

class Agent_QA():
    def __init__(self, persist_dir="./output", process=True, input_path=None,
                 embedding_model="BAAI/bge-base-en-v1.5", max_length=2048, retrieval_type="dense",
                 document_store=None, top_k=1, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5},
                 append=True, index_name="elastic_index_1", append_path=None,
                 response_template = "Please reformat your query to regenerate the answer.",
                 asset_path="/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets",):
        self.model = None
        self.tokenizer = None
        self.retrieval_type = retrieval_type
        self.retriever = None
        self.append_path = append_path
        self.intent_detector = IntentDetector()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.response_template = response_template
        self.search_type = search_type
        
        if os.path.exists(input_path):
            self.input_path = input_path
        elif os.path.exists(os.path.split(os.path.split(os.path.split(script_dir)[0])[0])[0] \
                            + '/assets/docs/'):
            self.input_path = os.path.split(os.path.split(os.path.split(script_dir)[0])[0])[0] \
                            + '/assets/docs/'
        elif os.path.exists(os.path.join(asset_path, 'docs/')):
            self.input_path = os.path.join(asset_path, 'docs/')
        else:
            print("The given file path is unavailable, please check and try again!")
        if append:
            self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type, document_store=document_store,
                                               persist_dir=persist_dir, process=process,
                                               embedding_model=embedding_model, max_length=max_length,
                                               index_name = index_name)
            self.db = self.doc_parser.KB_construct(self.input_path)
        else:
            print("Make sure the current persist path is new!")
            if os.path.exists(persist_dir):
                if bool(os.listdir(persist_dir)):
                    self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type,
                                                   document_store=document_store,
                                                   persist_dir=persist_dir, process=process,
                                                   embedding_model=embedding_model,
                                                   max_length=max_length,
                                                   index_name=index_name)
                    self.db = self.doc_parser.load(self.input_path)
                else:
                    self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type,
                                                       document_store=document_store,
                                                       persist_dir=persist_dir, process=process,
                                                       embedding_model=embedding_model, max_length=max_length,
                                                       index_name = index_name)
                    self.db = self.doc_parser.KB_construct(self.input_path)
            else:
                    self.doc_parser = DocumentIndexing(retrieval_type=self.retrieval_type,
                                                       document_store=document_store,
                                                       persist_dir=persist_dir, process=process,
                                                       embedding_model=embedding_model, max_length=max_length,
                                                       index_name = index_name)
                    self.db = self.doc_parser.KB_construct(self.input_path)
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=self.db, top_k=top_k,
                                   search_type=search_type, search_kwargs=search_kwargs)


    def append_localdb(self, 
                       append_path, 
                       top_k=1, 
                       search_type="similarity_score_threshold", 
                       search_kwargs={"score_threshold": 0.9, "k": 1}):
        self.db = self.doc_parser.KB_append(append_path)
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=self.db, top_k=top_k,
                           search_type=search_type, search_kwargs=search_kwargs)


    def pre_llm_inference_actions(self, model_name, query):
        intent = self.intent_detector.intent_detection(model_name, query)
        links = []
        docs = []
        if 'qa' not in intent.lower():
            print("Chat with AI Agent.")
            prompt = generate_prompt(query)
        else:
            print("Chat with QA agent.")
            if self.retriever:
                context, links = self.retriever.get_context(query)
                if len(context) == 0:
                    return "Response with template.", links
                if self.search_type == "similarity_score_threshold":
                    prompt = generate_qa_enterprise(query, context)
                else:
                    prompt = generate_qa_prompt(query, context)
            else:
                prompt = generate_prompt(query)
        return prompt, links