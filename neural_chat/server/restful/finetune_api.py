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


class FinetuneAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, chatbot) -> None:
        self.chatbot = chatbot

    def get_chatbot(self):
        if self.chatbot is None:
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot
    
    def handle_finetune_request(self) -> str:
        bot = self.get_chatbot()
        try:
            bot.finetune_model()
        except:
            raise Exception("Exception occurred when finetuning model, please check the arguments.")
        else:
            logger.info('Model finetuning finished.')
            return "Succeed"


router = FinetuneAPIRouter()


@router.post("/v1/finetune")
async def finetune_endpoint(request: str) -> str:
    return await router.handle_finetune_request(request)