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

from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse
# pylint: disable=E0611
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter
from ...cli.log import logger
from ...server.restful.openai_protocol import ChatCompletionRequest, ChatCompletionResponse
from ...config import GenerationConfig
import json
from ...plugins import plugins, is_plugin_enabled

def check_completion_request(request: BaseModel) -> Optional[str]:
    logger.info(f"Checking parameters of completion request...")
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

    def __init__(self) -> None:
        super().__init__()

    def set_chatbot(self, chatbot) -> None:
        self.chatbot = chatbot

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot


    def handle_completion_request(self, request: ChatCompletionRequest):
        return self.handle_chat_completion_request(request)


    def handle_chat_completion_request(self, request: ChatCompletionRequest):
        chatbot = self.get_chatbot()

        try:
            logger.info(f"Predicting chat completion using prompt '{request.prompt}'")
            config = GenerationConfig()
            # Set attributes of the config object from the request
            for attr, value in request.__dict__.items():
                if attr == "stream":
                    continue
                setattr(config, attr, value)
            response = ""
            if request.stream:
                generator, link = chatbot.predict_stream(query=request.prompt, config=config)
                def stream_generator():
                    for output in generator:
                        ret = {
                            "text": output,
                            "error_code": 0,
                        }
                        response += output + ' '
                        yield json.dumps(ret).encode() + b"\0"
                    yield f"data: [DONE]\n\n"
                if is_plugin_enabled("cache"):
                    plugins["cache"]["instance"].post_llm_inference_actions(request.prompt, response)
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                response = chatbot.predict(query=request.prompt, config=config)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        else:
            logger.info(f"Chat completion finished.")
            return ChatCompletionResponse(response=response)


router = TextChatAPIRouter()


@router.post("/v1/completions")
async def completion_endpoint(request: ChatCompletionRequest):
    ret = check_completion_request(request)
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    return router.handle_completion_request(request)


@router.post("/v1/chat/completions")
async def chat_completion_endpoint(chat_request: ChatCompletionRequest):
    ret = check_completion_request(chat_request)
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    return router.handle_chat_completion_request(chat_request)

@router.post("/v1/models")
async def show_available_models():
    models = []
    models.append(router.get_chatbot().model_name)
    return {"models": models}
