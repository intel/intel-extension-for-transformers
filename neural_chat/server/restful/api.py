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
from neural_chat.cli.log import logger


from .textchat_api import router as textchat_router
from .voicechat_api import router as voicechat_router
from .retrieval_api import router as retrieval_router
from .text2image_api import router as text2image_router
from .finetune_api import router as finetune_router

_router = APIRouter()


def setup_router(api_list: List):
    """setup router for fastapi

    Args:
        api_list (List): [textchat, voicechat, retrieval, text2image, finetune]

    Returns:
        APIRouter
    """
    for api_name in api_list:
        if api_name.lower() == 'textchat':
            _router.include_router(textchat_router)
        elif api_name.lower() == 'voicechat':
            _router.include_router(voicechat_router)
        elif api_name.lower() == 'retrieval':
            _router.include_router(retrieval_router)
        elif api_name.lower() == 'text2image':
            _router.include_router(text2image_router)
        elif api_name.lower() == 'finetune':
            _router.include_router(finetune_router)
        else:
            logger.error(f"NeuralChat has not supported such service yet: {api_name}")
            sys.exit(-1)

    return _router