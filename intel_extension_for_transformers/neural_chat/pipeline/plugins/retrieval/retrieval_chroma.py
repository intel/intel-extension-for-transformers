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

"""The wrapper for Chroma retriever based on langchain"""

from langchain.vectorstores.chroma import Chroma

class ChromaRetriever():
    """Retrieve the document database with Chroma database using dense retrieval."""

    def __init__(self, database=None, search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}):
        assert database is not None, "Please give a document database for retrieving."
        self.retriever = database.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def query_the_database(self, query):
        documents = self.retriever.get_relevant_documents(query)
        context = ""
        links = []
        for doc in documents: 
            context = context + doc.page_content + " "
            links.append(doc.metadata)
        return context.strip(), links
