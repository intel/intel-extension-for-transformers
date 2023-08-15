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

from typing import Optional
from fastapi import APIRouter
from neural_chat.cli.log import logger
from neural_chat.server.restful.request import FinetuneRequest
from neural_chat.server.restful.response import FinetuneResponse
from neural_chat.config import NeuralChatConfig
from neural_chat.chatbot import build_chatbot


def check_finetune_params(request: FinetuneRequest) -> Optional[str]:
    if request.content is not None and not isinstance(request.content, str):
        return f'Param Error: request.content {request.content} is not in the type of str'

    return None


class FinetuneAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, chatbot) -> None:
        self.chatbot = chatbot

    def get_chatbot(self):
        if self.chatbot is None:
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot
    
    def handle_finetune_request(self, request: FinetuneRequest) -> FinetuneResponse:
        bot = self.get_chatbot()
        result = bot.finetune_model(bot.config.finetune_config)
        return FinetuneResponse(content=result)


router = FinetuneAPIRouter()
config = NeuralChatConfig()
bot = build_chatbot(config)
router.set_chatbot(bot)


@router.post("/v1/finetune")
async def finetune_endpoint(request: FinetuneRequest) -> FinetuneResponse:
    ret = check_finetune_params(request)
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    return await router.handle_finetune_request(request)