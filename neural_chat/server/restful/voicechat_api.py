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

from typing import ByteString
from fastapi import APIRouter
from neural_chat.cli.log import logger
from fastapi import File, UploadFile, Form
from pydub import AudioSegment

class VoiceChatAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, chatbot) -> None:
        self.chatbot = chatbot

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot
    
    async def handle_voice_chat_request(self, file: UploadFile, voice: str) -> str:
        chatbot = self.get_chatbot()
        try:
            result = chatbot.chat(query=file.filename)
        except:
            raise Exception("Exception occurred when transfering voice to text.")
        else:
            logger.info('Chatbot inferencing finished.')
            return result


router = VoiceChatAPIRouter()

# voice to text
@router.post("/v1/voicechat/completions")
async def voicechat(file: UploadFile = File(...), voice: str = Form(...)):
    file_name = file.filename
    logger.info(f'Received file: {file_name}, and use voice: {voice}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_audio_bytes")
    audio.export(f"{file_name}", format="wav")
    return await router.handle_voice_chat_request(file_name, voice)
