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

"""The class definition for the retriever. Supporting langchain-based retriever."""

# from .retrieval_bm25 import SparseBM25Retriever
from .retrieval_chroma import ChromaRetriever
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)
class Retriever():
    """The wrapper for sparse retriever and dense retriever."""

    def __init__(self, retrieval_type="dense", document_store=None,
                 top_k=1, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}):
        self.retrieval_type = retrieval_type
        if not document_store:
            raise ValueError(f"Please give a knowledge base for retrieval.")
        if retrieval_type == "dense":
            self.retriever = ChromaRetriever(database=document_store,
                                             search_type=search_type,
                                             search_kwargs=search_kwargs)
        else:
            # self.retriever = SparseBM25Retriever(document_store=document_store, top_k=top_k)
            ### Will be removed in another PR
            logging.info("This vector database will be removed in another PR.")
    def get_context(self, query):
        context, links = self.retriever.query_the_database(query)
        return context, links
