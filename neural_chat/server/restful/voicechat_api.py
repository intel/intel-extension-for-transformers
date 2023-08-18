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
from typing import Optional
from neural_chat.cli.log import logger
from fastapi import File, UploadFile, Form
from pydub import AudioSegment
from neural_chat.config import GenerationConfig

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
    
    async def handle_voice_chat_request(self, filename: str, audio_output_path: Optional[str]=None) -> str:
        chatbot = self.get_chatbot()
        try:
            config = GenerationConfig(max_new_tokens=64, audio_output_path=audio_output_path)
            result = chatbot.chat(query=filename, config=config)
        except Exception as e:
            raise Exception(e)
        else:
            logger.info('Voice chatbot inferencing finished.')
            return result


router = VoiceChatAPIRouter()

# voice to text
@router.post("/v1/voicechat/completions")
async def voicechat(file: UploadFile=File(...), voice: str=Form(...), audio_output_path: str=Form(...)):
    file_name = file.filename
    logger.info(f'Received file: {file_name}, and use voice: {voice}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_audio_bytes")
    audio.export(f"{file_name}", format="wav")
    if audio_output_path is not " ":
        logger.info(f'Predicting voicechat with audio output, output path is {audio_output_path}')
        return await router.handle_voice_chat_request(file_name, audio_output_path)
    else:
        logger.info(f'Predicting voicechat with text output.')
        return await router.handle_voice_chat_request(file_name)
    
