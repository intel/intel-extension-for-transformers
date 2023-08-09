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
from neural_chat.server.restful.openai_protocal import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice, 
    ChatCompletionRequest, ChatCompletionResponseChoice, ChatCompletionResponse, 
    UsageInfo, ModelCard, ModelList, ModelPermission, ChatMessage
)


# TODO: process request and return params in Dict
def generate_params(request: CompletionRequest) -> Dict:
    return {}


def check_completion_request(request: BaseModel) -> Optional[str]:
    if request.max_tokens is not None and request.max_tokens <= 0:
        return f"Param Error: {request.max_tokens} is less than the minimum of 1 --- 'max_tokens'"

    if request.n is not None and request.n <= 0:
        return f"Param Error: {request.n} is less than the minimum of 1 --- 'n'"
    
    if request.temperature is not None and request.temperature < 0:
        return f"Param Error: {request.temperature} is less than the minimum of 0 --- 'temperature'"
    
    if request.temperature is not None and request.temperature > 2:
        return f"Param Error: {request.temperature} is greater than the maximum of 2 --- 'temperature'",

    if request.top_p is not None and request.top_p < 0:
        return f"Param Error: {request.top_p} is less than the minimum of 0 --- 'top_p'",

    if request.top_p is not None and request.top_p > 1:
        return f"Param Error: {request.top_p} is greater than the maximum of 1 --- 'top_p'",

    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, )
    ):
        return f"Param Error: {request.stop} is not valid under any of the given schemas --- 'stop'",

    return None


class TextChatAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, chatbot: Chatbot) -> None:
        self.chatbot = chatbot

    def get_chatbot(self) -> Chatbot:
        if self.chatbot is None:
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot
    
    def get_model_list(self) -> List[str]:
        chatbot = self.get_chatbot()
        model_list = chatbot.get_model_list()
        if model_list is None or len(model_list)==0:
            raise RuntimeError("No Model list is found.")
        return model_list
    
    async def handle_models_request(self) -> ModelList:
        model_list = self.get_model_list()
        model_list.sort()
        model_cards = []
        for m in model_list:
            model_cards.append(ModelCard(id=m, root=m, permission=[ModelPermission()]))
        return ModelList(data=model_cards)

    # TODO: process stream chat completion
    # TODO: add log
    async def handle_completion_request(self, request:CompletionRequest) -> CompletionResponse:
        chatbot = self.get_chatbot()
        params = generate_params(request)
        if request.stream:
            # TODO: process stream chat completion
            inference_results = chatbot.predict_stream(params)
        else:
            inference_results = chatbot.predict(params)
        
        choices = []
        usage = UsageInfo()
        for i, content in enumerate(inference_results):
            choices.append(
                CompletionResponseChoice(
                    index=i,
                    text=content["text"],
                    logprobs=content.get("logprobs", None),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            task_usage = UsageInfo.parse_obj(content["usage"])
            for usage_key, usage_value in task_usage.dict().items():
                setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo.parse_obj(usage)
        )
    
    # TODO: process stream chat completion
    # TODO: add log
    async def handle_chat_completion_request(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        chatbot = self.get_chatbot()
        params = generate_params(request)
        if request.stream:
            # TODO: process stream chat completion
            return ChatCompletionResponse()

        choices = []
        chat_completions = []
        usage = UsageInfo()
        for i in range(request.n):
            content = chatbot.predict(params)
            chat_completions.append(content)
        try:
            all_tasks = await asyncio.gather(*chat_completions)
        except Exception as e:
            return e
        usage = UsageInfo()
        for i, content in enumerate(all_tasks):
            if content["error_code"] != 0:
                return f'Error {content["error_code"]}: content["text"]'
            choices.append(
                ChatCompletionResponseChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=content["text"]),
                    finish_reason=content.get("finish_reason", "stop"),
                )
            )
            if "usage" in content:
                task_usage = UsageInfo.parse_obj(content["usage"])
                for usage_key, usage_value in task_usage.dict().items():
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

        return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)
    

router = TextChatAPIRouter()


@router.post("/v1/models")
async def models_endpoint() -> ModelList:
    return await router.handle_models_request()

    
@router.post("/v1/completion")
async def completion_endpoint(request: CompletionRequest) -> CompletionResponse:
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