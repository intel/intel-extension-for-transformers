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
from ...cli.log import logger
from collections.abc import Iterable
from ...server.restful.request import TGIRequest
from starlette.responses import StreamingResponse
from huggingface_hub import InferenceClient


class TextGenerationAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.endpoint = "http://0.0.0.0:9876/"
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

    def handle_tgi_request(self, prompt, parameters, stream=False):
        client = InferenceClient(model=self.endpoint)
        best_of = parameters.get("best_of", 1)
        do_sample = parameters.get("do_sample", True)
        max_new_tokens = parameters.get("max_new_tokens", 20)
        repetition_penalty = parameters.get("repetition_penalty", 1.03)
        temperature = parameters.get("temperature", 0.5)
        top_k = parameters.get("top_k", 10)
        top_p = parameters.get("top_p", 0.95)
        typical_p = parameters.get("typical_p", 0.95)
        res = client.text_generation(
            prompt=prompt, best_of=best_of, do_sample=do_sample,
            max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
            temperature=temperature, top_k=top_k, top_p=top_p,
            typical_p=typical_p, stream=stream)
        return res

    def handle_tgi_generate_request(self, request: TGIRequest):
        logger.info(f"[tgi] request data: {request}")

        try:
            response = self.handle_tgi_request(
                prompt=request.inputs,
                parameters=request.parameters,
                stream=False
            )
        except Exception as e:
            logger.exception(f"[tgi] Exception occurred. {e}")
            raise Exception(f"Exception occurred. {e}")
        else:
            logger.info(f"[tgi] Inferencing succeed. Result: {response}")
            return response


    def handle_tgi_generate_stream_request(self, request: TGIRequest):
        logger.info(f"[tgi] request data: {request}")

        try:
            response_stream = self.handle_tgi_request(
                prompt=request.inputs,
                parameters=request.parameters,
                stream=True
            )
            def stream_generator():
                if isinstance(response_stream, Iterable):
                    # pylint: disable=not-an-iterable
                    for output in response_stream:
                        yield f"data: {output}\n\n"
                else:
                    yield f"data: {response_stream}\n\n"
                yield f"data: [DONE]\n\n"
            logger.info(f"[tgi] Streaming response")
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        except Exception as e:
            logger.exception(f"[tgi] Exception occurred. {e}")
            raise Exception(f"Exception occurred. {e}")


router = TextGenerationAPIRouter()


@router.post("/v1/tgi")
async def tgi_root(request: TGIRequest):
    return router.handle_tgi_generate_request(request)


@router.post("/v1/tgi/generate")
async def tgi_generate(request: TGIRequest):
    return router.handle_tgi_generate_request(request)


@router.post("/v1/tgi/generate_stream")
async def tgi_generate_stream(request: TGIRequest):
    return router.handle_tgi_generate_stream_request(request)
