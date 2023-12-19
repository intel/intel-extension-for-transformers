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
from langchain.vectorstores.chroma import Chroma as Chroma_origin
_DEFAULT_PERSIST_DIR = './output'
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


class Chroma(Chroma_origin):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        sign: Optional[str] = None,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = _DEFAULT_PERSIST_DIR,
        client_settings: Optional[chromadb.config.Settings] = None,
        client: Optional[chromadb.Client] = None,  # Add this line
        collection_metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if 'doc_id' in metadatas[0]: 
            ids = [doc.metadata['doc_id'] for doc in documents]
        if sign == 'child':
            persist_directory = persist_directory + "_child"
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )


    @classmethod
    def build(
            cls: Type[Chroma],
            documents: List[Document],
            sign: Optional[str] = None,
            embedding: Optional[Embeddings] = None,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            persist_directory: Optional[str] = None,
            client_settings: Optional[chromadb.config.Settings] = None,
            client: Optional[chromadb.Client] = None,
            collection_metadata: Optional[Dict] = None,
            **kwargs: Any,
    ) -> Chroma:
        if not persist_directory:
            persist_directory = _DEFAULT_PERSIST_DIR
        if sign == 'child':
            persist_directory = persist_directory + "_child"
        if os.path.exists(persist_directory):
            if bool(os.listdir(persist_directory)):
                print("Load the existing database!")
                chroma_collection = cls(
                    collection_name=collection_name,
                    embedding_function=embedding,
                    persist_directory=persist_directory,
                    client_settings=client_settings,
                    client=client,
                    collection_metadata=collection_metadata,
                    **kwargs,
                )
                return chroma_collection
        else:
            print("Create a new kb...")
            chroma_collection = cls.from_documents(
                content=content,
                embedding=embedding,
                ids=ids,
                collection_name=collection_name,
                persist_directory=persist_directory,
                client_settings=client_settings,
                client=client,
                collection_metadata=collection_metadata,
                **kwargs,
            )
            return chroma_collection
            

    @classmethod
    def reload(
            cls: Type[Chroma],
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            embedding_function: Optional[Embeddings] = None,
            persist_directory: Optional[str] = None,
            client_settings: Optional[chromadb.config.Settings] = None,
            collection_metadata: Optional[Dict] = None,
            client: Optional[chromadb.Client] = None,
            relevance_score_fn: Optional[Callable[[float], float]] = None,
    ) -> Chroma:
        
        if not persist_directory:
            persist_directory = _DEFAULT_PERSIST_DIR
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
        )
        return chroma_collection
    
