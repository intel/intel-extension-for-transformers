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

from fastapi import APIRouter
# pylint: disable=E0611
from pydantic import BaseModel
from typing import Optional
from intel_extension_for_transformers.neural_chat.cli.log import logger
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
from transformers import TrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.server.restful.request import FinetuneRequest


def check_finetune_request(request: BaseModel) -> Optional[str]:
    logger.info(f"Checking parameters of finetune request...")
    if request.train_file is None and request.dataset_name is None:
        return f"Param Error: finetune dataset can not be None"
    return None


class FinetuneAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, chatbot, use_deepspeed, world_size, host, port) -> None:
        self.chatbot = chatbot
        self.use_deepspeed = use_deepspeed
        self.world_size = world_size
        self.host = host
        self.port = port

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot

    def handle_finetune_request(self, request: FinetuneRequest) -> str:
        try:
            model_args = ModelArguments(model_name_or_path=request.model_name_or_path)
            data_args = DataArguments(train_file=request.train_file,
                                      dataset_name=request.dataset_name,
                                      dataset_concatenation=request.dataset_concatenation)
            training_args = TrainingArguments(
                output_dir=request.output_dir,
                do_train=True,
                max_steps=request.max_steps,
                overwrite_output_dir=request.overwrite_output_dir
            )
            finetune_args = FinetuningArguments(peft=request.peft)
            finetune_cfg = TextGenerationFinetuningConfig(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                finetune_args=finetune_args,
            )
            finetune_model(finetune_cfg)
        except Exception as e:
            raise Exception(e)
        else:
            logger.info('Model finetuning finished.')
            return "Succeed"


router = FinetuneAPIRouter()


@router.post("/v1/finetune")
async def finetune_endpoint(request: FinetuneRequest):
    ret = check_finetune_request(request)
    if ret is not None:
        raise RuntimeError(f"Invalid parameter: {ret}")
    return router.handle_finetune_request(request)
