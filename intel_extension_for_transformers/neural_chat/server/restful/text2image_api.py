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
from typing import ByteString
from ...cli.log import logger
from ...server.restful.response import ImageResponse


class Text2ImageAPIRouter(APIRouter):

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

    async def handle_text2image_request(self, request: str) -> ImageResponse:
        chatbot = self.get_chatbot()
        try:
            image = chatbot.predict(request)
        except:
            raise Exception("Exception occurred when generating image from text.")
        else:
            logger.info('Text transferring to image finished.')
            return ImageResponse(image=image, response="Succeed")


router = Text2ImageAPIRouter()

@router.post("/v1/text2image")
async def text2image(request: str) -> str:
    return await router.handle_text2image_request(request)
