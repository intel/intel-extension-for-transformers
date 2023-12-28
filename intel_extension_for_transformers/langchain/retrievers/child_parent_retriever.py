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

"""The wrapper for Child-Parent retriever based on langchain"""
from langchain.retrievers import MultiVectorRetriever
from langchain_core.vectorstores import VectorStore
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from enum import Enum

class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class ChildParentRetriever(MultiVectorRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""
    
    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    parentstore: VectorStore
    
    def get_context(self, query:str, *, run_manager: CallbackManagerForRetrieverRun):
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            The concatation of the retrieved documents and the link
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        
        ids = []
        for d in sub_docs:
            if d.metadata['doc_id'] not in ids:
                ids.append(d.metadata['doc_id'])
        retrieved_documents = self.parentstore.get(ids)
        context = ''
        links = []
        for doc in retrieved_documents:
            context = context + doc.page_content + " "
            links.append(doc.metadata['source'])
        return context.strip(), links
