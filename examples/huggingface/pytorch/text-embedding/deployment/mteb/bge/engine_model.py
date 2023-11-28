from typing import cast, List, Dict, Union

import numpy as np
import torch
from mteb import DRESModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import copy


class EngineBGEModel(DRESModel):
    def __init__(self,
                 model_name_or_path: str = None,
                 pooling_method: str = 'cls',
                 normalize_embeddings: bool = True,
                 query_instruction_for_retrieval: str = None,
                 batch_size: int = 256,
                 backend: str = 'Engine',
                 **kwargs) -> None:

        ort_model_path = kwargs.get("ort_model_path", None)
        engine_model = kwargs.get("engine_model", None)
        self.backend = kwargs.get("backend", 'Engine')

        if backend == 'Engine':
            self.engine_model = engine_model.graph
            file_name = kwargs.get("file_name", None)
            print('The backend is Neural Engine, evaluate on: ', ort_model_path, file_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.pytorch_model = AutoModel.from_pretrained(model_name_or_path)
            self.hidden_size = self.pytorch_model.config.hidden_size
        elif backend == 'Pytorch':
            print('The backend is Pytorch, evaluate on: ', ort_model_path, file_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.pytorch_model = AutoModel.from_pretrained(model_name_or_path)
        elif backend == 'Onnxruntime':
            print('The backend is Onnxruntime.')
            file_name = kwargs.get("file_name", None)
            print('The backend is Onnxruntime, evaluate on: ', ort_model_path, file_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.ort_model = ORTModelForFeatureExtraction.from_pretrained(ort_model_path, file_name=file_name)

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.pytorch_model is not None:
            self.pytorch_model = self.pytorch_model.to(self.device)

            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                self.pytorch_model = torch.nn.DataParallel(self.pytorch_model)
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
        if self.backend == 'Engine':
            ort_all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size),
                                    desc="Batches",
                                    disable=len(sentences) < 256):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(self.device)

                ort_inputs = self.tokenizer(sentences_batch,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="np")

                input_ids = np.ascontiguousarray(ort_inputs['input_ids'])
                token_type_ids = np.ascontiguousarray(ort_inputs['token_type_ids'])
                attention_mask = np.ascontiguousarray(ort_inputs['attention_mask'])

                engine_input = [input_ids, token_type_ids, attention_mask]
                result = copy.deepcopy(self.engine_model.inference(engine_input))
                ort_last_hidden_state = torch.tensor(result['last_hidden_state:0']).reshape(
                    input_ids.shape[0], input_ids.shape[1], self.hidden_size)
                ort_embeddings = self.pooling(ort_last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    ort_embeddings = torch.nn.functional.normalize(ort_embeddings, dim=-1)
                ort_embeddings = cast(torch.Tensor, ort_embeddings)
                ort_all_embeddings.append(ort_embeddings.cpu().numpy())
            return np.concatenate(ort_all_embeddings, axis=0)
        elif self.backend == 'Pytorch':
            self.pytorch_model.eval()
            all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size),
                                    desc="Batches",
                                    disable=len(sentences) < 256):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(self.device)
                ort_inputs = self.tokenizer(sentences_batch,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="np")
                last_hidden_state = self.pytorch_model(**inputs, return_dict=True).last_hidden_state
                embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                embeddings = cast(torch.Tensor, embeddings)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.concatenate(all_embeddings, axis=0)

        elif self.backend == 'Onnxruntime':
            ort_all_embeddings = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size),
                                    desc="Batches",
                                    disable=len(sentences) < 256):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                inputs = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(self.device)

                ort_inputs = self.tokenizer(sentences_batch,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="np")

                ort_last_hidden_state = torch.tensor(self.ort_model(**ort_inputs).last_hidden_state)
                ort_embeddings = self.pooling(ort_last_hidden_state, inputs['attention_mask'])
                if self.normalize_embeddings:
                    ort_embeddings = torch.nn.functional.normalize(ort_embeddings, dim=-1)
                ort_embeddings = cast(torch.Tensor, ort_embeddings)
                ort_all_embeddings.append(ort_embeddings.cpu().numpy())
            return np.concatenate(ort_all_embeddings, axis=0)

    def pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
