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
from __future__ import annotations
import base64
import logging, os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
import numpy as np
from langchain_core.documents import Document
from langchain.vectorstores.chroma import Chroma as Chroma_origin
from langchain_core.embeddings import Embeddings
from langchain_core.utils import xor_args
from langchain_core.vectorstores import VectorStore
import chromadb
import chromadb.config
_DEFAULT_PERSIST_DIR = './output'
_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)


class Chroma(Chroma_origin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_texts(
        cls: Type[Chroma],
        texts: List[str],
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
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts (List[str]): List of texts to add to the collection.
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
        )
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        if hasattr(
            chroma_collection._client, "max_batch_size"
        ):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=chroma_collection._client,
                ids=ids,
                metadatas=metadatas,
                documents=texts,
            ):
                chroma_collection.add_texts(
                    texts=batch[3] if batch[3] else [],
                    metadatas=batch[2] if batch[2] else None,
                    ids=batch[0],
                )
        else:
            chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        sign: str = None,
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
        if sign == "child":
            persist_directory = persist_directory + "_child"
        if os.path.exists(persist_directory):
            if bool(os.listdir(persist_directory)):
                logging.info("Load the existing database!")
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
            logging.info("Create a new knowledge base...")
            chroma_collection = cls.from_documents(
                documents=documents,
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
            embedding: Optional[Embeddings] = None,
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
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
        )
        return chroma_collection
