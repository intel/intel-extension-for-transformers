from typing import List, Optional, Union

import torch
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document

import fastrag
from fastrag.stores import PLAIDDocumentStore

logger = fastrag.utils.init_logger(__name__)


class ColBERTRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: PLAIDDocumentStore,
        use_gpu: bool = True,
        top_k: int = 10,
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        super().__init__()  # TODO: Not sure I need this

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        self.document_store = document_store
        self.use_gpu = use_gpu
        self.top_k = top_k

        logger.info(f"Init retriever using the store: {document_store}")

    def retrieve(
        self, query: str, top_k: Optional[int] = None, filters=None, **kwargs
    ) -> List[Document]:
        if filters:
            logger.info(f"Filters are not implemented for ColBERT/PLAID.")

        if top_k is None:
            top_k = self.top_k

        documents = self.document_store.query(query_str=query, top_k=top_k)

        return documents

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        scale_score: bool = None,
        filters=None,
        **kwargs,
    ) -> List[List[Document]]:
        if filters:
            logger.info(f"Filters are not implemented for ColBERT/PLAID.")

        if top_k is None:
            top_k = self.top_k

        documents = self.document_store.query_batch(queries, top_k)

        return documents