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
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1 import Field
from enum import Enum
from typing import List
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class ChildParentRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""
    vectorstore: VectorStore
    parentstore: VectorStore
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            The concatation of the retrieved documents and the link
        """
        ids = []
        results = []

        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        for d in sub_docs:
            if d.metadata["identify_id"] not in ids:
                ids.append(d.metadata['identify_id'])

        retrieved_documents = self.parentstore.get(ids)
        for i in range(len(retrieved_documents)):
            metadata = retrieved_documents['metadatas'][i]
            context = retrieved_documents['documents'][i]
            instance = Document(page_content=context, metadata=metadata)
            results.append(instance)

        return results
