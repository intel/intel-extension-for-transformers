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
import logging
from typing import Any, Type, List, Optional, TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.qdrant import Qdrant as Qdrant_origin
from intel_extension_for_transformers.transformers.utils.utility import LazyImport

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

if TYPE_CHECKING:
    from qdrant_client.conversions import common_types

_DEFAULT_PERSIST_DIR = './output'

qdrant_client = LazyImport("qdrant_client")

class Qdrant(Qdrant_origin):

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        sign: Optional[str] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str]= None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        force_recreate: Optional[bool] = False,
        **kwargs: Any,
    ):
        """Create a Qdrant vectorstore from a list of documents.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): A subclass of `Embeddings`, responsible for text vectorization.
            sign (Optional[str], optional): sign for retrieval_type of 'child_parent'. Defaults to None.
            location (Optional[str], optional):
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
                Defaults to None.
            url (Optional[str], optional): either host or str of "Optional[scheme], host, Optional[port],
                Optional[prefix]". Defaults to None.
            api_key (Optional[str], optional): API key for authentication in Qdrant Cloud. Defaults to None.
            host (Optional[str], optional): Host name of Qdrant service. If url and host are None, set to
                'localhost'. Defaults to None.
            persist_directory (Optional[str], optional): Path in which the vectors will be stored while using
                local mode. Defaults to None.
            collection_name (Optional[str], optional): Name of the Qdrant collection to be used.
                Defaults to _LANGCHAIN_DEFAULT_COLLECTION_NAME.
            force_recreate (bool, optional): _description_. Defaults to False.
        """
        if sum([param is not None for param in (location, url, host, persist_directory)]) == 0:
            # One of 'location', 'url', 'host' or 'persist_directory' should be specified.
            persist_directory = _DEFAULT_PERSIST_DIR
            if sign == "child":
                persist_directory = persist_directory + "_child"
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts,
            embedding,
            metadatas=metadatas,
            location=location,
            url=url,
            api_key=api_key,
            host=host,
            path=persist_directory,
            collection_name=collection_name,
            force_recreate=force_recreate,
            **kwargs)

    @classmethod
    def build(
        cls,
        documents: List[Document],
        embedding: Optional[Embeddings],
        sign: Optional[str] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str]= None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        force_recreate: Optional[bool] = False,
        **kwargs: Any,
    ):
        """Build a Qdrant vectorstore.

        Args:
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): A subclass of `Embeddings`, responsible for text vectorization.
            sign (Optional[str], optional): sign for retrieval_type of 'child_parent'. Defaults to None.
            location (Optional[str], optional):
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
                Defaults to None.
            url (Optional[str], optional): either host or str of "Optional[scheme], host, Optional[port],
                Optional[prefix]". Defaults to None.
            api_key (Optional[str], optional): API key for authentication in Qdrant Cloud. Defaults to None.
            host (Optional[str], optional): Host name of Qdrant service. If url and host are None, set to
                'localhost'. Defaults to None.
            persist_directory (Optional[str], optional): Path in which the vectors will be stored while using
                local mode. Defaults to None.
            collection_name (Optional[str], optional): Name of the Qdrant collection to be used.
                Defaults to _LANGCHAIN_DEFAULT_COLLECTION_NAME.
            force_recreate (bool, optional): _description_. Defaults to False.
            kwargs:
                Current used:
                    port (Optional[int], optional): Port of the REST API interface. Defaults to 6333.
                    grpc_port (int, optional): Port of the gRPC interface. Defaults to 6334.
                    prefer_grpc (bool, optional): If true - use gPRC interface whenever possible in custom methods.
                        Defaults to False.
                    https (Optional[bool], optional): If true - use HTTPS(SSL) protocol.
                    prefix (Optional[str], optional):
                        If not None - add prefix to the REST URL path.
                        Example: service/v1 will result in
                            http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                    timeout (Optional[float], optional):
                        Timeout for REST and gRPC API requests.

                    distance_func (str, optional): Distance function. One of: "Cosine" / "Euclid" / "Dot".
                        Defaults to "Cosine".
                    content_payload_key (str, optional): A payload key used to store the content of the document.
                        Defaults to CONTENT_KEY.
                    metadata_payload_key (str, optional): A payload key used to store the metadata of the document.
                        Defaults to METADATA_KEY.
                    vector_name (Optional[str], optional): Name of the vector to be used internally in Qdrant.
                        Defaults to VECTOR_NAME.
                    shard_number (Optional[int], optional): Number of shards in collection.
                    replication_factor (Optional[int], optional):
                        Replication factor for collection.
                        Defines how many copies of each shard will be created.
                        Have effect only in distributed mode.
                    write_consistency_factor (Optional[int], optional):
                        Write consistency factor for collection.
                        Defines how many replicas should apply the operation for us to consider
                        it successful. Increasing this number will make the collection more
                        resilient to inconsistencies, but will also make it fail if not enough
                        replicas are available.
                        Does not have any performance impact.
                        Have effect only in distributed mode.
                    on_disk_payload (Optional[bool], optional):
                        If true - point`s payload will not be stored in memory.
                        It will be read from the disk every time it is requested.
                        This setting saves RAM by (slightly) increasing the response time.
                        Note: those payload values that are involved in filtering and are
                        indexed - remain in RAM.
                    hnsw_config (Optional[common_types.HnswConfigDiff], optional): Params for HNSW index.
                    optimizers_config (Optional[common_types.OptimizersConfigDiff], optional): Params for optimizer.
                    wal_config (Optional[common_types.WalConfigDiff], optional): Params for Write-Ahead-Log.
                    quantization_config (Optional[common_types.QuantizationConfig], optional):
                        Params for quantization, if None - quantization will be disable.
                    init_from (Optional[common_types.InitFrom], optional):
                        Use data stored in another collection to initialize this collection.
                    on_disk (Optional[bool], optional): if True, vectors will be stored on disk.
                        If None, default value will be used.
        """
        if sum([param is not None for param in (location, url, host, persist_directory)]) == 0:
            # One of 'location', 'url', 'host' or 'persist_directory' should be specified.
            persist_directory = _DEFAULT_PERSIST_DIR
            if sign == "child":
                persist_directory = persist_directory + "_child"
        if persist_directory and os.path.exists(persist_directory):
            if bool(os.listdir(persist_directory)):
                logging.info("Load the existing database!")
                texts = [d.page_content for d in documents]
                qdrant_collection = cls.construct_instance(
                    texts=texts,
                    embedding=embedding,
                    location=location,
                    url=url,
                    api_key=api_key,
                    host=host,
                    path=persist_directory,
                    collection_name=collection_name,
                    force_recreate=force_recreate,
                    **kwargs
                )
                return qdrant_collection
        else:
            logging.info("Create a new knowledge base...")
            qdrant_collection = cls.from_documents(
                documents=documents,
                embedding=embedding,
                location=location,
                url=url,
                api_key=api_key,
                host=host,
                persist_directory=persist_directory,
                collection_name=collection_name,
                force_recreate=force_recreate,
                **kwargs,
            )
            return qdrant_collection


    @classmethod
    def reload(
        cls,
        embedding: Optional[Embeddings],
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str]= None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        force_recreate: bool = False,
        **kwargs: Any,
    ):
        """Reload a Qdrant vectorstore.

        Args:
            embedding (Optional[Embeddings]): A subclass of `Embeddings`, responsible for text vectorization.
            location (Optional[str], optional):
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
                Defaults to None.
            url (Optional[str], optional): either host or str of "Optional[scheme], host, Optional[port],
                Optional[prefix]". Defaults to None.
            api_key (Optional[str], optional): API key for authentication in Qdrant Cloud. Defaults to None.
            host (Optional[str], optional): Host name of Qdrant service. If url and host are None, set to
                'localhost'. Defaults to None.
            persist_directory (Optional[str], optional): Path in which the vectors will be stored while using
                local mode. Defaults to None.
            collection_name (Optional[str], optional): Name of the Qdrant collection to be used.
                Defaults to _LANGCHAIN_DEFAULT_COLLECTION_NAME.
            force_recreate (bool, optional): _description_. Defaults to False.
        """
        if sum([param is not None for param in (location, url, host, persist_directory)]) == 0:
            # One of 'location', 'url', 'host' or 'persist_directory' should be specified.
            persist_directory = _DEFAULT_PERSIST_DIR

        # for a single quick embedding to get vector size
        tmp_texts = ["foo"]

        qdrant_collection = cls.construct_instance(
            texts=tmp_texts,
            embedding=embedding,
            location=location,
            url=url,
            api_key=api_key,
            host=host,
            path=persist_directory,
            collection_name=collection_name,
            force_recreate=force_recreate,
            **kwargs
        )
        return qdrant_collection


    def is_local(
        self,
    ):
        """Determine whether a client is local."""
        if hasattr(self.client, "_client") and \
            isinstance(self.client._client, qdrant_client.local.qdrant_local.QdrantLocal):
            return True
        else:
            return False


    @classmethod
    def _document_from_scored_point(
        cls,
        scored_point: Any,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        metadata = scored_point.payload.get(metadata_payload_key) or {}
        metadata["_id"] = scored_point.id
        # bug in langchain_core v0.1.18
        # comment out the calling to collection_name with bug "ScoredPoint object has no attribute collection_name"
        # bug was fixed in https://github.com/langchain-ai/langchain/pull/16920.
        # TODO: remove this func after release of langchain_core version v0.1.19
        # metadata["_collection_name"] = scored_point.collection_name
        return Document(
            page_content=scored_point.payload.get(content_payload_key),
            metadata=metadata,
        )
