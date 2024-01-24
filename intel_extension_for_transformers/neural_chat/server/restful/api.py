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

import sys
from typing import List

from fastapi import APIRouter
from ...cli.log import logger

from .textchat_api import router as textchat_router
from .voicechat_api import router as voicechat_router
from .retrieval_api import router as retrieval_router
from .text2image_api import router as text2image_router
from .finetune_api import router as finetune_router
from .faceanimation_api import router as faceanimation_router
from .photoai_api import router as photoai_router
from .plugin_audio_api import router as plugin_audio_router
from .plugin_image2image_api import router as plugin_image2image_router
from .codegen_api import router as codegen_router
from .tgi_api import router as tgi_router

_router = APIRouter()

# Create a dictionary to map API names to their corresponding routers
api_router_mapping = {
    'textchat': textchat_router,
    'voicechat': voicechat_router,
    'retrieval': retrieval_router,
    'text2image': text2image_router,
    'finetune': finetune_router,
    'faceanimation': faceanimation_router,
    'photoai': photoai_router,
    'plugin_audio': plugin_audio_router,
    "image2image": plugin_image2image_router,
    'codegen': codegen_router,
    'tgi': tgi_router
}

def setup_router(api_list, chatbot=None, enable_llm=True, use_deepspeed=False, world_size=1, host="0.0.0.0", port=80):
    """Setup router for FastAPI

    Args:
        api_list (List): List of API names
        chatbot: The chatbot instance

    Returns:
        APIRouter
    """
    for api_name in api_list:
        lower_api_name = api_name.lower()
        if lower_api_name in api_router_mapping:
            api_router = api_router_mapping[lower_api_name]
            if enable_llm:
                api_router.set_chatbot(chatbot, use_deepspeed, world_size, host, port)
            if lower_api_name == "plugin_image2image":
                api_router.worker.start()
                logger.info("create main worker done...")
            _router.include_router(api_router)
        else:
            logger.error(f"NeuralChat has not supported such service yet: {api_name}")
            sys.exit(-1)

    return _router
