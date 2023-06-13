from pathlib import Path
from typing import List, Optional, Union

import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.search.strided_tensor_core import StridedTensorCore
from haystack.modeling.utils import initialize_device_settings
from haystack.nodes.ranker import BaseRanker
from haystack.schema import Document


class ColBERTRanker(BaseRanker):
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        top_k: int = 10,
        query_maxlen: int = 120,
        doc_maxlen: int = 120,
        dim: int = 128,
        use_gpu: bool = False,
        devices: Optional[List[Union[int, str, torch.device]]] = None,
    ):
        self.top_k = top_k

        if devices is not None:
            self.devices = devices
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        self.config = ColBERTConfig(
            ncells=2,
            nbits=2,
            query_maxlen=query_maxlen,
            doc_maxlen=doc_maxlen,
            dim=dim,
        )

        self.use_gpu = use_gpu
        self.to_cpu = not use_gpu

        self.checkpoint = Checkpoint(checkpoint_path, self.config)
        if self.use_gpu:
            self.checkpoint.model.cuda()

        self.checkpoint.query_tokenizer.query_maxlen = query_maxlen
        self.checkpoint.query_tokenizer.doc_maxlen = doc_maxlen

    def predict(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if top_k is None:
            top_k = self.top_k

        query_emb = self.checkpoint.queryFromText([query], bsize=64, to_cpu=self.to_cpu)
        if self.use_gpu:
            query_emb = query_emb.half()

        tensor, lengths = self.checkpoint.docFromText(
            [d.content for d in documents], keep_dims="flatten", bsize=64, to_cpu=self.to_cpu
        )
        tensor, masks = StridedTensorCore(tensor, lengths, use_gpu=self.use_gpu).as_padded_tensor()
        if not self.use_gpu:
            tensor, masks = tensor.float(), masks.float()

        # MaxSim implementation. Returns [#queries, #documents] tensor.
        # Output identical to colbert.modeling.colbert.colbert_score
        # however scores are different than when calculated at query time.
        # It's due to quantization differences when recovering the embeddings.
        scores = torch.einsum("bwd,qvd -> bqwv", query_emb, tensor * masks).max(-1).values.sum(-1)

        indices = scores.cpu().sort(descending=True).indices[0]
        return [documents[i.item()] for i in indices[:top_k]]

    def predict_batch(
        self,
        queries: List[str],
        documents: Union[List[Document], List[List[Document]]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[List[Document]]:
        if top_k is None:
            top_k = self.top_k

        predictions = []

        if isinstance(documents[0], list):
            assert len(queries) == len(
                documents
            ), "Lists of documents should match number of queries."
            for query, q_documents in zip(queries, documents):
                predictions.append(self.predict(query, q_documents, top_k))

        else:
            q_embeddings = self.checkpoint.queryFromText(queries, bsize=64, to_cpu=self.to_cpu)
            if self.use_gpu:
                q_embeddings = q_embeddings.half()

            tensor, lengths = self.checkpoint.docFromText(
                [d.content for d in documents], keep_dims="flatten", bsize=64
            )
            tensor, masks = StridedTensorCore(
                tensor, lengths, use_gpu=self.use_gpu
            ).as_padded_tensor()

            if not self.use_gpu:
                tensor, masks = tensor.float(), masks.float()

            scores = (
                torch.einsum("bwd,qvd -> bqwv", q_embeddings, tensor * masks).max(-1).values.sum(-1)
            )
            indices = scores.cpu().sort(1, descending=True).indices
            for q_index in indices:
                predictions.append([documents[i.item()] for i in q_index[:top_k]])

        return predictions