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

"""The wrapper for Retriever based on langchain"""
from intel_extension_for_transformers.langchain.retrievers import VectorStoreRetriever, ChildParentRetriever
from langchain.storage import InMemoryByteStore
import logging

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

class RetrieverAdapter():
    """Retrieve the document database with Chroma database using dense retrieval."""

    def __init__(self, retrieval_type='default', document_store=None, child_document_store= None, **kwargs):
        self.retrieval_type = retrieval_type
        
        if self.retrieval_type == "default":
            self.retriever = VectorStoreRetriever(vectorstore = document_store, **kwargs)
        elif self.retrieval_type == "child_parent":
            store = InMemoryByteStore()
            self.retriever = ChildParentRetriever(vectorstore=child_document_store, parentstore=document_store, \
                                                  byte_store=store, **kwargs)
        else:
            logging.error('The chosen retrieval type remains outside the supported scope.')
