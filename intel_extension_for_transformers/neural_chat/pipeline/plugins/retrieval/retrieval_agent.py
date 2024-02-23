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
from typing import Dict, List, Any, ClassVar, Collection
from .detector.intent_detection import IntentDetector
from .detector.query_explainer import QueryPolisher
from .parser.parser import DocumentParser
from .retriever_adapter import RetrieverAdapter
from intel_extension_for_transformers.neural_chat.pipeline.plugins.prompt.prompt_template \
    import generate_qa_prompt, generate_prompt, generate_qa_enterprise
from intel_extension_for_transformers.langchain.embeddings import HuggingFaceEmbeddings, \
    HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from intel_extension_for_transformers.langchain.vectorstores import Chroma, Qdrant
import uuid
from langchain_core.documents import Document
import logging

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

def document_transfer(data_collection):
    "Transfer the raw document into langchain supported format."
    documents = []
    for data, meta in data_collection:
        doc_id = str(uuid.uuid4())
        metadata = {"source": meta, "identify_id":doc_id}
        doc = Document(page_content=data, metadata=metadata)
        documents.append(doc)
    return documents

def document_append_id(documents):
    for _doc in documents:
        _doc.metadata["doc_id"] = _doc.metadata["identify_id"]
    return documents


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
                 mode = "accuracy",
                 process=True,
                 append=True,
                 polish=False,
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
        self.splitter = RecursiveCharacterTextSplitter(chunk_size= kwargs['child_size'] \
                    if 'child_size' in kwargs else 512)
        allowed_retrieval_type: ClassVar[Collection[str]] = (
            "default",
            "child_parent",
        )
        allowed_generation_mode: ClassVar[Collection[str]] = (
            "accuracy",
            "general",
        )
        if polish:
            self.polisher = QueryPolisher()
        else:
            self.polisher = None

        assert self.retrieval_type in allowed_retrieval_type, "search_type of {} not allowed.".format(   \
            self.retrieval_type)
        assert self.mode in allowed_generation_mode, "generation mode of {} not allowed.".format( \
            self.mode)

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
            logging.error("The given file path is unavailable, please check and try again!")
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
            logging.error("Please select a proper embedding model.")
            logging.error(e)

        self.document_parser = DocumentParser(max_chuck_size=max_chuck_size, min_chuck_size = min_chuck_size, \
                                              process=self.process)
        data_collection = self.document_parser.load(input=self.input_path, **kwargs)
        logging.info("The parsing for the uploaded files is finished.")

        langchain_documents = document_transfer(data_collection)
        logging.info("The format of parsed documents is transferred.")

        if self.vector_database == "Chroma":
            self.database = Chroma
        elif self.vector_database == "Qdrant":
            self.database = Qdrant
        # elif self.vector_database == "PGVector":
        #     self.database = PGVector()

        if self.retrieval_type == 'default':  # Using vector store retriever
            if append:
                knowledge_base = self.database.from_documents(documents=langchain_documents, embedding=self.embeddings,
                                                              **kwargs)
            else:
                knowledge_base = self.database.build(documents=langchain_documents, embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              **kwargs)
            if self.vector_database == "Qdrant" and knowledge_base.is_local():
               # one local storage folder cannot be accessed by multiple instances of Qdrant client simultaneously.
               knowledge_base.client.close()
        elif self.retrieval_type == "child_parent":    # Using child-parent store retriever
            child_documents = self.splitter.split_documents(langchain_documents)
            langchain_documents = document_append_id(langchain_documents)
            if append:
                knowledge_base = self.database.from_documents(documents=langchain_documents, embedding=self.embeddings,
                                                              **kwargs)
                child_knowledge_base = self.database.from_documents(documents=child_documents, sign='child', \
                                                                    embedding=self.embeddings, **kwargs)
            else:
                knowledge_base = self.database.build(documents=langchain_documents, embedding=self.embeddings, **kwargs)
                child_knowledge_base = self.database.build(documents=langchain_documents, embedding=self.embeddings, \
                                            sign='child', **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                               child_document_store=child_knowledge_base, **kwargs)
            if self.vector_database == "Qdrant" :
                # one local storage folder cannot be accessed by multiple instances of Qdrant client simultaneously.
                if knowledge_base.is_local():
                    knowledge_base.client.close()
                if child_knowledge_base.is_local():
                    child_knowledge_base.client.close()
        elif self.retrieval_type == "bm25":
            self.docs = document_append_id(langchain_documents)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, docs=self.docs, **kwargs)
        logging.info("The retriever is successfully built.")

    def reload_localdb(self, local_persist_dir, **kwargs):
        """
        Reload the local existed knowledge base. Do not support inmemory database.
        """
        assert os.path.exists(local_persist_dir) and bool(os.listdir(local_persist_dir)), \
            "Please check the local knowledge base was built!"
        assert self.retrieval_type != "bm25", "Do not support inmemory database."

        knowledge_base = self.database.reload(persist_directory=local_persist_dir, embedding=self.embeddings, **kwargs)
        if self.retrieval_type == 'default':
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              **kwargs)
        elif self.retrieval_type == "child_parent":
            child_persist_dir = local_persist_dir + "_child"
            child_knowledge_base = self.database.reload(persist_directory=child_persist_dir, \
                                                        embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              child_document_store=child_knowledge_base, **kwargs)

        logging.info("The retriever is successfully built.")


    def create(self, input_path, **kwargs):
        """
        Create a new knowledge base based on the uploaded files.
        """
        data_collection = self.document_parser.load(input=input_path, **kwargs)
        langchain_documents = document_transfer(data_collection)

        if self.retrieval_type == 'default':
            knowledge_base = self.database.from_documents(documents=langchain_documents, \
                                                          embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              **kwargs)
        elif self.retrieval_type == "child_parent":
            child_documents = self.splitter.split_documents(langchain_documents)
            langchain_documents = document_append_id(langchain_documents)
            knowledge_base = self.database.from_documents(documents=langchain_documents, \
                                                          embedding=self.embeddings, **kwargs)
            child_knowledge_base = self.database.from_documents(documents=child_documents, sign='child', \
                                                                embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              child_document_store=child_knowledge_base, **kwargs)
        elif self.retrieval_type == "bm25":
            self.docs = document_append_id(langchain_documents)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, docs=self.docs, **kwargs)
        logging.info("The retriever is successfully built.")


    def append_localdb(self, append_path, **kwargs):
        "Append the knowledge instances into a given knowledge base."

        data_collection = self.document_parser.load(input=append_path, **kwargs)
        langchain_documents = document_transfer(data_collection)

        if self.retrieval_type == 'default':
            knowledge_base = self.database.from_documents(documents=langchain_documents, \
                                                          embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, \
                                              document_store=knowledge_base, **kwargs)
        elif self.retrieval_type == "child_parent":
            child_documents = self.splitter.split_documents(langchain_documents)
            langchain_documents = document_append_id(langchain_documents)
            knowledge_base = self.database.from_documents(documents=langchain_documents, \
                                                          embedding=self.embeddings, **kwargs)
            child_knowledge_base = self.database.from_documents(documents=child_documents, sign = 'child', \
                                                          embedding=self.embeddings, **kwargs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, document_store=knowledge_base, \
                                              child_document_store=child_knowledge_base, **kwargs)
        elif self.retrieval_type == "bm25":
            new_docs = document_append_id(langchain_documents)
            self.docs = self.docs.extend(new_docs)
            self.retriever = RetrieverAdapter(retrieval_type=self.retrieval_type, docs=self.docs, **kwargs)
        logging.info("The retriever is successfully built.")



    def pre_llm_inference_actions(self, model_name, query):
        if self.polisher:
            try:
                query = self.polisher.polish_query(model_name, query)
            except Exception as e:
                logging.info(f"Polish the user query failed, {e}")
                raise Exception("[Rereieval ERROR] query polish failed!")

        try:
            intent = self.intent_detector.intent_detection(model_name, query)
        except Exception as e:
            logging.info(f"intent detection failed, {e}")
            raise Exception("[Rereieval ERROR] intent detection failed!")
        links = []
        context = ''
        assert self.retriever is not None, logging.info("Please check the status of retriever")
        if self.mode == "accuracy":
        # "retrieval with threshold" will only return the document that bigger than the threshold.
            context, links = self.retriever.get_context(query)
            if 'qa' not in intent.lower() and context == '':
                logging.info("Chat with AI Agent.")
                prompt = generate_prompt(query)
            else:
                logging.info("Chat with QA Agent.")
                if len(context) == 0:
                    return "Response with template.", links
                prompt = generate_qa_enterprise(query, context)
        elif self.mode == "general":
        # For general setting, will return top-k documents.
            if 'qa' not in intent.lower() and context == '':
                logging.info("Chat with AI Agent.")
                prompt = generate_prompt(query)
            else:
                logging.info("Chat with QA Agent.")
                context, links = self.retriever.get_context(query)
                if len(context) == 0:
                    return "Response with template.", links
                prompt = generate_qa_prompt(query, context)
        else:
            logging.error("The selected generation mode is invalid!")
        return prompt, links
