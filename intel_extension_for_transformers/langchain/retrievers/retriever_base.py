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

class Retriever():
    """Retrieve the document database with Chroma database using dense retrieval."""

    def __init__(self, retrieval_type='default', document_store=None, **kwargs):
        self.retrieval_type = retrieval_type
        
        if self.retrieval_type == "default":
            self.retriever = self.vector_as_retriever(document_store = document_store, **kwargs)
        else:
            print("Other types of retriever will coming soon.")
        
    
    def vector_as_retriever(self, document_store, **kwargs):
        search_type = ''
        search_kwargs = ''
        if 'search_type' in kwargs and 'search_kwargs' in kwargs:
            search_type = kwargs['search_type']
            search_kwargs = kwargs['search_kwargs'] 
        else:
            search_type = 'mmr'
            search_kwargs = {"k": 1, "fetch_k": 5}
        return document_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    def get_context(self, query):
        if self.retrieval_type == 'default':
            context = ''
            links = []
            retrieved_documents = self.retriever.get_relevant_documents(query)
            for doc in retrieved_documents:
                context = context + doc.page_content + " "
                links.append(doc.metadata)
            return context.strip(), links
        else:
            print("Other types of retriever will coming soon.")