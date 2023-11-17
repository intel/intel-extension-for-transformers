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
"""Wrapper for parsing the uploaded user file and then make document indexing."""

import os
from langchain.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, \
    HuggingFaceBgeEmbeddings, GooglePalmEmbeddings
from .context_utils import load_unstructured_data, laod_structured_data, get_chuck_data


class DocumentIndexing:
    def __init__(self, retrieval_type="dense", document_store=None, persist_dir="./output",
                 process=True, embedding_model="BAAI/bge-base-en-v1.5", max_length=512,
                 index_name=None):
        """
        Wrapper for document indexing. Support dense and sparse indexing method.
        """
        self.retrieval_type = retrieval_type
        self.document_store = document_store
        self.process = process
        self.persist_dir = persist_dir
        self.max_length = max_length
        self.index_name = index_name
        
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
            
        
        
    def parse_document(self, input):
        """
        Parse the uploaded file.
        """
        if input.endswith("pdf") or input.endswith("docx") or input.endswith("html") \
           or input.endswith("txt") or input.endswith("md"):
            content = load_unstructured_data(input)
            if self.process:
                chuck = get_chuck_data(content, self.max_length, input)
            else:
                chuck = [[content.strip(),input]]
        elif input.endswith("jsonl") or input.endswith("xlsx") or input.endswith("csv"):
            chuck = laod_structured_data(input, self.process, self.max_length)
        else:
            print("This file is ignored. Will support this file format soon.")
        return chuck


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
                        chuck = get_chuck_data(content, self.max_length, input)
                    else:
                        chuck = [[content.strip(),input]]
                    paragraphs += chuck
                elif filename.endswith("jsonl") or filename.endswith("xlsx") or input.endswith("csv"):
                    chuck = laod_structured_data(os.path.join(dirpath, filename), self.process, self.max_length)
                    paragraphs += chuck
                else:
                    print("This file {} is ignored. Will support this file format soon.".format(filename))
        return paragraphs
    
    def load(self, input):
        if self.retrieval_type=="dense":
            vectordb = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        else:
            # if self.document_store == "inmemory":
            #     vectordb = self.KB_construct(input)
            # else:
            #     vectordb = ElasticsearchDocumentStore(host="localhost", index=self.index_name,
            #                                           port=9200, search_fields=["content", "title"])
            vectordb=None
            print("will be removed in another PR")
        return vectordb
            
    def KB_construct(self, input):
        """
        Construct the local knowledge base based on the uploaded file/files.
        """
        if self.retrieval_type == "dense":
            if os.path.exists(input):
                if os.path.isfile(input):
                    data_collection = self.parse_document(input)
                elif os.path.isdir(input):
                    data_collection = self.batch_parse_document(input)
                else:
                    print("Please check your upload file and try again!")
                    
                documents = []
                for data, meta in data_collection:
                    if len(data) < 5:
                        continue
                    metadata = {"source": meta}
                    new_doc = Document(page_content=data, metadata=metadata)
                    documents.append(new_doc)
                assert documents!= [], "The given file/files cannot be loaded." 
                vectordb = Chroma.from_documents(documents=documents, embedding=self.embeddings,
                                                 persist_directory=self.persist_dir)
                vectordb.persist()
                print("The local knowledge base has been successfully built!")
                return vectordb
            else:
                print("There might be some errors, please wait and try again!")
        else:
            # if os.path.exists(input):
            #     if os.path.isfile(input):
            #         data_collection = self.parse_document(input)
            #     elif os.path.isdir(input):
            #         data_collection = self.batch_parse_document(input)
            #     else:
            #         print("Please check your upload file and try again!")
            #     if self.document_store == "inmemory":
            #         document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)
            #     elif self.document_store == "Elasticsearch":
            #         document_store = ElasticsearchDocumentStore(host="localhost", index=self.index_name,
            #                                                     port=9200, search_fields=["content", "title"])
            # 
            #     documents = []
            #     for data, meta in data_collection:
            #         metadata = {"source": meta}
            #         if len(data) < 5:
            #             continue
            #         new_doc = SDocument(content=data, meta=metadata)
            #         documents.append(new_doc)
            #     assert documents != [], "The given file/files cannot be loaded."
            #     document_store.write_documents(documents)
            #     print("The local knowledge base has been successfully built!")
            #     return document_store
            # else:
            #     print("There might be some errors, please wait and try again!")
            print("Will be removed in another PR")


    def KB_append(self, input):  ### inmemory documentstore please use KB construct
        if self.retrieval_type == "dense":
            if os.path.exists(input):
                if os.path.isfile(input):
                    data_collection = self.parse_document(input)
                elif os.path.isdir(input):
                    data_collection = self.batch_parse_document(input)
                else:
                    print("Please check your upload file and try again!")

                documents = []
                for data, meta in data_collection:
                    if len(data) < 5:
                        continue
                    metadata = {"source": meta}
                    new_doc = Document(page_content=data, metadata=metadata)
                    documents.append(new_doc)
                assert documents != [], "The given file/files cannot be loaded."
                embedding = HuggingFaceInstructEmbeddings(model_name=self.embedding_model)
                vectordb = Chroma.from_documents(documents=documents, embedding=embedding,
                                                 persist_directory=self.persist_dir)
                vectordb.persist()
                print("The local knowledge base has been successfully built!")
                return Chroma(persist_directory=self.persist_dir, embedding_function=embedding)
            else:
                print("There might be some errors, please wait and try again!")
        else:
            # if os.path.exists(input):
            #     if os.path.isfile(input):
            #         data_collection = self.parse_document(input)
            #     elif os.path.isdir(input):
            #         data_collection = self.batch_parse_document(input)
            #     else:
            #         print("Please check your upload file and try again!")
            # 
            #     if self.document_store == "Elasticsearch":
            #         document_store = ElasticsearchDocumentStore(host="localhost", index=self.index_name,
            #                                                     port=9200, search_fields=["content", "title"])
            #         documents = []
            #         for data, meta in data_collection:
            #             metadata = {"source": meta}
            #             if len(data) < 5:
            #                 continue
            #             new_doc = SDocument(content=data, meta=metadata)
            #             documents.append(new_doc)
            #         assert documents != [], "The given file/files cannot be loaded."
            #         document_store.write_documents(documents)
            #         print("The local knowledge base has been successfully built!")
            #         return ElasticsearchDocumentStore(host="localhost", index=self.index_name,
            #                                                   port=9200, search_fields=["content", "title"])
            #     else:
            #         print("Unsupported document store type, please change to Elasticsearch!")
            # else:
            #     print("There might be some errors, please wait and try again!")
            print("Will be removed in another PR.")
