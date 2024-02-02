#!/usr/bin/env python
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

from typing import Optional, Dict
# pylint: disable=E0611
from pydantic import BaseModel, Extra


class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class RetrievalRequest(RequestBaseModel):
    query: str
    domain: str
    blob: Optional[str]
    filename: Optional[str]
    embedding: Optional[str] = 'dense'
    params: Optional[Dict] = None
    debug: Optional[bool] = False
    retrieval_type: str
    document_path: str


class FinetuneRequest(RequestBaseModel):
    model_name_or_path: str = "Intel/neural-chat-7b-v3-1"
    train_file: str = None
    dataset_name: str = None
    output_dir: str = './tmp'
    max_steps: int = 3
    overwrite_output_dir: bool = True
    dataset_concatenation: bool = False
    peft: str = 'lora'


class AskDocRequest(RequestBaseModel):
    query: str
    translated: str
    domain: str
    blob: Optional[str]
    filename: Optional[str]
    knowledge_base_id: Optional[str] = 'default'
    embedding: Optional[str] = 'dense'
    params: Optional[dict] = None
    debug: Optional[bool] = False
    stream: bool = True
    max_new_tokens: int = 256


class FeedbackRequest(RequestBaseModel):
    """
    Request class for feedback api
    'feedback_id': set to be auto_increment, no need to pass as argument
    'feedback': 0 for 'like', 1 for 'dislike'
    """
    # feedback_id: Optional[int] = None
    question: str
    answer: str
    feedback: Optional[int] = 0
    comments: Optional[str] = ""


class TGIRequest(RequestBaseModel):
    inputs: str
    parameters: Optional[dict] = None
    stream: Optional[bool] = False
