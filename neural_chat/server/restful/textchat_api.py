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

import asyncio
from fastapi.routing import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi import APIRouter
from neural_chat.cli.log import logger
from neural_chat.server.restful.openai_protocol import ChatCompletionRequest, ChatCompletionResponse


def check_completion_request(request: BaseModel) -> Optional[str]:
    if request.temperature is not None and request.temperature < 0:
        return f"Param Error: {request.temperature} is less than the minimum of 0 --- 'temperature'"
    
    if request.temperature is not None and request.temperature > 2:
        return f"Param Error: {request.temperature} is greater than the maximum of 2 --- 'temperature'"

    if request.top_p is not None and request.top_p < 0:
        return f"Param Error: {request.top_p} is less than the minimum of 0 --- 'top_p'"

    if request.top_p is not None and request.top_p > 1:
        return f"Param Error: {request.top_p} is greater than the maximum of 1 --- 'top_p'"

    if request.top_k is not None and (not isinstance(request.top_k, int)):
        return f"Param Error: {request.top_k} is not valid under any of the given schemas --- 'top_k'"

    if request.top_k is not None and request.top_k < 1:
        return f"Param Error: {request.top_k} is greater than the minimum of 1 --- 'top_k'"

    if request.max_new_tokens is not None and (not isinstance(request.max_new_tokens, int)):
        return f"Param Error: {request.max_new_tokens} is not valid under any of the given schemas --- 'max_new_tokens'"

    return None


class TextChatAPIRouter(APIRouter):

    def __init__(self, chatbot) -> None:
        super().__init__()

    def set_chatbot(self, chatbot) -> None:
        self.chatbot = chatbot

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot
    

    async def handle_completion_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return await self.handle_chat_completion_request(request)
    

    async def handle_chat_completion_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        chatbot = self.get_chatbot()

        try:
            if request.stream:
                response = chatbot.predict_stream(query=request.prompt)
            else:
                response = chatbot.predict(query=request.prompt)
        except Exception:
            raise Exception("Exception occurred while chat completion.")
        else:
            return ChatCompletionResponse(response=response) 
    

router = TextChatAPIRouter()

    
@router.post("/v1/completions")
async def completion_endpoint(request: ChatCompletionRequest) -> ChatCompletionResponse:
    ret = check_completion_request()
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    return await router.handle_completion_request(request)


@router.post("/v1/chat/completions")
async def chat_completion_endpoint(chat_request: ChatCompletionRequest) -> ChatCompletionResponse:
    ret = check_completion_request()
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    return await router.handle_chat_completion_request(chat_request)
