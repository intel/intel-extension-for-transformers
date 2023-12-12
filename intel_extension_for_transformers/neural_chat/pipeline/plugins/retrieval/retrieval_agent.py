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
from typing import Dict, List, Any
from .detector.intent_detection import IntentDetector
from .parser.parser import DocumentParser
from intel_extension_for_transformers.neural_chat.pipeline.plugins.prompt.prompt_template \
    import generate_qa_prompt, generate_prompt, generate_qa_enterprise
from intel_extension_for_transformers.langchain.embeddings import HuggingFaceEmbeddings, \
    HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from intel_extension_for_transformers.langchain.retrievers import Retriever
from intel_extension_for_transformers.langchain.vectorstores import Chroma

class Agent_QA():
    """
    The agent for retrieval-based chatbot. Contains all parameters setting and file parsing.
    """
    def __init__(self,
                 vector_database="Chroma",
                 embedding_model="BAAI/bge-base-en-v1.5",
                 input_path = None,
                 response_template="Please reformat your query to regenerate the answer.",
                 asset_path="/intel-extension-for-transformers/intel_extension_for_transformers/neural_chat/assets",
                 retrieval_type = 'default',
                 max_chuck_size=512,
                 min_chuck_size=5,
                 mode = 1,
                 process=True,
                 append=True,
                 **kwargs):

        self.intent_detector = IntentDetector()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.response_template = response_template
        self.vector_database = vector_database
        self.input_path = None
        self.retrieval_type = retrieval_type
        self.mode = mode
        self.process = process
        self.retriever = None

        if isinstance(input_path, str):
            if os.path.exists(input_path):
                self.input_path = input_path
            elif os.path.exists(os.path.split(os.path.split(os.path.split(script_dir)[0])[0])[0] \
                                + '/assets/docs/'):
                self.input_path = os.path.split(os.path.split(os.path.split(script_dir)[0])[0])[0] \
                                + '/assets/docs/'
            elif os.path.exists(os.path.join(asset_path, 'docs/')):
                self.input_path = os.path.join(asset_path, 'docs/')
        elif isinstance(input_path, List):
            self.input_path = input_path
        else:
            print("The given file path is unavailable, please check and try again!")
        assert self.input_path != None, "Should gave an input path!"

        try:
            if "instruct" in embedding_model:
                self.embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model)
            elif "bge" in embedding_model:
                self.embeddings = HuggingFaceBgeEmbeddings(
                    model_name=embedding_model,
                    encode_kwargs={'normalize_embeddings': True},
                    query_instruction="Represent this sentence for searching relevant passages:")
            elif "Google" == embedding_model:
                self.embeddings = GooglePalmEmbeddings()
            else:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    encode_kwargs={"normalize_embeddings": True},
                )
        except Exception as e:
            print("Please selet a proper embedding model")
        
        self.document_parser = DocumentParser(max_chuck_size=max_chuck_size, min_chuck_size = min_chuck_size, \
                                              process=self.process)
        data_collection = self.document_parser.load(input=self.input_path, **kwargs)

        if self.vector_database == "Chroma":
            self.database = Chroma()
        # elif self.vector_database == "PGVector":
        #     self.database = PGVector()

        if append:
            knowledge_base = self.database.from_collection(content=data_collection, embedding=self.embeddings, **kwargs)
        else:
            knowledge_base = self.database.build(content=data_collection, embedding=self.embeddings, **kwargs)
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=knowledge_base, **kwargs)


    def reload_localdb(self, local_persist_dir, **kwargs):
        """
        Reload the local existed knowledge base.
        """
        assert os.path.exists(local_persist_dir) and bool(os.listdir(local_persist_dir)), \
            "Please check the local knowledge base was built!"
        knowledge_base = self.database.reload(persist_directory=local_persist_dir, **kwargs)
        if "retrieval_type" in kwargs:
            self.retrieval_type = kwargs['retrieval_type']
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=knowledge_base, **kwargs)


    def create(self, input_path, **kwargs):
        """
        Create a new knowledge base based on the uploaded files.
        """
        data_collection = self.document_parser.load(input=input_path, **kwargs)
        knowledge_base = self.database.from_collection(content=data_collection, **kwargs)
        if "retrieval_type" in kwargs:
            self.retrieval_type = kwargs['retrieval_type']
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=knowledge_base, **kwargs)


    def append_localdb(self, append_path, **kwargs):
        "Append the knowledge instances into a given knowledge base."

        data_collection = self.document_parser.load(input=append_path, **kwargs)
        if "retrieval_type" in kwargs:
            self.retrieval_type = kwargs['retrieval_type']
        knowledge_base = self.database.from_collection(content=data_collection, **kwargs)
        self.retriever = Retriever(retrieval_type=self.retrieval_type, document_store=knowledge_base, **kwargs)


    def pre_llm_inference_actions(self, model_name, query):
        intent = self.intent_detector.intent_detection(model_name, query)
        links = []
        context = ''
        assert self.retriever is not None, print("Please check the status of retriever")
        if self.mode == 1:   ## "retrieval with threshold" will only return the document that bigger than the threshold.
            context, links = self.retriever.get_context(query)
            if 'qa' not in intent.lower() and context == '':
                print("Chat with AI Agent.")
                prompt = generate_prompt(query)
            else:
                print("Chat with QA Agent.")
                if len(context) == 0:
                    return "Response with template.", links
                prompt = generate_qa_enterprise(query, context)
        elif self.mode == 2: ## For general setting, will return top-k documents.
            if 'qa' not in intent.lower() and context == '':
                print("Chat with AI Agent.")
                prompt = generate_prompt(query)
            else:
                print("Chat with QA Agent.")
                context, links = self.retriever.get_context(query)
                if len(context) == 0:
                    return "Response with template.", links
                prompt = generate_qa_prompt(query, context)
        return prompt, links
