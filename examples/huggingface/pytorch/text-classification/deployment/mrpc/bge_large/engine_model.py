from typing import cast, List, Dict, Union

import numpy as np
import torch
from mteb import DRESModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction


class EngineBGEModel(DRESModel):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            batch_size: int = 256,
            **kwargs
    ) -> None:

        ort_model_path = kwargs.get("ort_model_path", None)
        self.model = None
        self.ort_model = None
        if ort_model_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
        else:
            file_name = kwargs.get("file_name", None)
            print('Evaluate on onnx model', file_name)
            self.tokenizer = AutoTokenizer.from_pretrained(ort_model_path)
            self.ort_model = ORTModelForFeatureExtraction.from_pretrained(ort_model_path, file_name=file_name)

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.model is not None:
            self.model = self.model.to(self.device)

            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.batch_size = self.batch_size * num_gpus


    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if self.query_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
        else:
            input_texts = queries
        return self.encode(input_texts)


    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)


    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        if self.model is not None:
            self.model.eval()

            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<256):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(self.device)
                ort_inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np"
                )
                last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)
        
        elif self.ort_model is not None:
            ort_all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<256):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(self.device)

                ort_inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np"
                )

                ort_last_hidden_state = torch.tensor(self.ort_model(**ort_inputs).last_hidden_state)
                ort_embeddings = self.pooling(ort_last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    ort_embeddings = torch.nn.functional.normalize(ort_embeddings, dim=-1)
                ort_embeddings = cast(torch.Tensor, ort_embeddings)
                ort_all_embeddings.append(ort_embeddings.cpu().numpy())
            return np.concatenate(ort_all_embeddings, axis=0)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d


