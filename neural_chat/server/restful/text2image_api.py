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

import base64
from io import BytesIO
from PIL import Image
from fastapi import APIRouter
from typing import Optional
from neural_chat.cli.log import logger
from neural_chat.server.restful.request import Text2ImageRequest
from neural_chat.server.restful.response import ImageResponse


def check_text2image_params(request: Text2ImageRequest) -> Optional[str]:
    if request.steps is not None and (not isinstance(request.steps, int)):
        return f'Param Error: request.steps {request.steps} is not in the type of int'
    
    if request.seed is not None and (not isinstance(request.seed, int)):
        return f'Param Error: request.seed {request.seed} is not in the type of int'
    
    if request.guidance_scale is not None and (
        not isinstance(request.guidance_scale, int) and not isinstance(request.guidance_scale, float)):
        return f'Param Error: request.guidance_scale {request.guidance_scale} is not valid under any of the given schemas'
    
    return None


class Text2ImageAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.sdbot = None

    def set_sdbot(self, sdbot: SDbot) -> None:
        self.sdbot = sdbot

    def get_sdbot(self) -> SDbot:
        if self.sdbot is None:
            raise RuntimeError("Chatbot instance has not been set.")
        return self.sdbot

    async def handle_text2image_request(self, request: Text2ImageRequest) -> ImageResponse:
        data = {
            "prompt": request.prompt,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "token": request.sd_inference_token
        }
        sdbot = self.get_sdbot()
        image_string = sdbot.predict(data)
        image_byte = base64.b64decode(image_string)
        image_io = BytesIO(image_byte)
        image = Image.open(image_io)
        return ImageResponse(image=image)
    

router = Text2ImageAPIRouter()

@router.post("/v1/text2image/inference")
async def text2image(request: Text2ImageRequest) -> ImageResponse:
    ret = check_text2image_params(request)
    if ret is not None:
        raise RuntimeError("Invalid parameter.")
    return await router.handle_text2image_request(request)
