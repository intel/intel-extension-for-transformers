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
from concurrent import futures
# pylint: disable=E0611
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter
import requests
from ...cli.log import logger
from ...server.restful.openai_protocol import ChatCompletionRequest, ChatCompletionResponse
from ...config import GenerationConfig
import json, types
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

    def is_generator(self, obj):
        return isinstance(obj, types.GeneratorType)

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
            if chatbot.device == "hpu":
                config.device = "hpu"
                config.use_hpu_graphs = True
                config.task = "chat"
            buffered_texts = ""
            if request.stream:
                generator, link = chatbot.predict_stream(query=request.prompt, config=config)
                if not self.is_generator(generator):
                    generator = (generator,)
                def stream_generator():
                    nonlocal buffered_texts
                    for output in generator:
                        if isinstance(output, str):
                            chunks = output.split()
                            for chunk in chunks:
                                ret = {
                                    "text": chunk,
                                    "error_code": 0,
                                }
                                buffered_texts += chunk + ' '
                                yield json.dumps(ret).encode() + b"\0"
                        else:
                            ret = {
                                "text": output,
                                "error_code": 0,
                            }
                            buffered_texts += output + ' '
                            yield json.dumps(ret).encode() + b"\0"
                    yield f"data: [DONE]\n\n"
                    if is_plugin_enabled("cache") and \
                       not plugins["cache"]["instance"].pre_llm_inference_actions(request.prompt):
                        plugins["cache"]["instance"].post_llm_inference_actions(request.prompt, buffered_texts)
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

# router /v1/code_generation only supports non-streaming mode.
@router.post("/v1/code_generation")
async def code_generation_endpoint(chat_request: ChatCompletionRequest):
    if router.use_deepspeed:
        responses = []

        def send_request(port):
            try:
                url = f'http://{router.host}:{port}/v1/code_generation'
                response = requests.post(url, json=chat_request.dict())
                response.raise_for_status()
                json_response = json.loads(response.content)
                chat_completion_response = ChatCompletionResponse(response=json_response['response'])
                responses.append(chat_completion_response)
            except requests.exceptions.RequestException as e:
                print(f"Error sending/receiving on port {port}: {e}")

        with futures.ThreadPoolExecutor(max_workers=router.world_size) as executor:
            worker_ports = [router.port + i + 1 for i in range(router.world_size)]
            executor.map(send_request, worker_ports)
        if responses:
            return responses[0]
    else:
        ret = check_completion_request(chat_request)
        if ret is not None:
            raise RuntimeError("Invalid parameter.")
        return router.handle_chat_completion_request(chat_request)

# router /v1/code_chat supports both non-streaming and streaming mode.
@router.post("/v1/code_chat")
async def code_chat_endpoint(chat_request: ChatCompletionRequest):
    if router.use_deepspeed:
        if chat_request.stream:
            responses = []
            def generate_stream(port):
                url = f'http://{router.host}:{port}/v1/code_generation'
                response = requests.post(url, json=chat_request.dict(), stream=True, timeout=1000)
                responses.append(response)
            with futures.ThreadPoolExecutor(max_workers=router.world_size) as executor:
                worker_ports = [router.port + i + 1 for i in range(router.world_size)]
                executor.map(generate_stream, worker_ports)

            while not responses:
                pass
            def generate():
                if responses[0]:
                    for chunk in responses[0].iter_lines(decode_unicode=False, delimiter=b"\0"):
                        if chunk:
                            data = chunk.decode("utf-8")
                            yield f"data: {data}\n\n"
                    yield f"data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            responses = []

            def send_request(port):
                try:
                    url = f'http://{router.host}:{port}/v1/code_generation'
                    response = requests.post(url, json=chat_request.dict())
                    response.raise_for_status()
                    json_response = json.loads(response.content)
                    chat_completion_response = ChatCompletionResponse(response=json_response['response'])
                    responses.append(chat_completion_response)
                except requests.exceptions.RequestException as e:
                    print(f"Error sending/receiving on port {port}: {e}")

            with futures.ThreadPoolExecutor(max_workers=router.world_size) as executor:
                worker_ports = [router.port + i + 1 for i in range(router.world_size)]
                executor.map(send_request, worker_ports)
            if responses:
                return responses[0]
    else:
        ret = check_completion_request(chat_request)
        if ret is not None:
            raise RuntimeError("Invalid parameter.")
        return router.handle_chat_completion_request(chat_request)
