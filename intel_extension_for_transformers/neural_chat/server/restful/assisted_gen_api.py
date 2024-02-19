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

import httpx
from fastapi.routing import APIRouter
from fastapi import APIRouter
from ...cli.log import logger
from .openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)


class AssistedGenerationAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()

    def set_chatbot(self, chatbot, use_deepspeed=False, world_size=1, host="0.0.0.0", port=80) -> None:
        self.chatbot = chatbot
        self.use_deepspeed = use_deepspeed
        self.world_size = world_size
        self.host = host
        self.port = port
        assistant_host = chatbot.assistant_host
        assistant_port = chatbot.assistant_port
        self.assistant_prefix = 'http://'+assistant_host+":"+assistant_port

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot

    async def handle_assist_chat(self, request: ChatCompletionRequest):
        async with httpx.AsyncClient() as client:
            response = await client.get(self.assistant_prefix+"/v1/assist/decode", params=request)
            return response.json()

    async def handle_assist_decode(self, request: ChatCompletionRequest):
        chatbot = self.get_chatbot()
        # TODO: complete model inferencing process for assisted model
        pass

    async def handle_assist_data_transfer(self, request: ChatCompletionRequest):
        async with httpx.AsyncClient() as client:
            response = await client.get(self.assistant_prefix+"/v1/assist/data_transfer", params=request)
            return response.json()


router = AssistedGenerationAPIRouter()


# router for small model to do inferencing
@router.post("/v1/assist/chat")
async def assist_chat(request: ChatCompletionRequest):
    return await router.handle_assist_chat(request)


# router for assisted model to do inferencing
@router.post("/v1/assist/decode")
async def assist_decode(request: CompletionRequest):
    return await router.handle_assist_decode(request)


# router for assisted model to do data transferring
@router.post("/v1/assist/data_transfer")
async def assist_data_transfer(request: CompletionRequest):
    return await router.handle_assist_data_transfer(request)

