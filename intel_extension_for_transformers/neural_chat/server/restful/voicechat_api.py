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

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import Optional
from ...cli.log import logger
from fastapi import File, UploadFile, Form
from pydub import AudioSegment
from ...config import GenerationConfig
from ...plugins import plugins
import base64
import torch

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

    def handle_voice_asr_request(self, filename: str) -> str:
        chatbot = self.get_chatbot()
        try:
            return chatbot.asr.audio2text(filename)
        except Exception as e:
            raise Exception(e)

    async def handle_voice_chat_request(self, prompt: str, voice: str, audio_output_path: Optional[str]=None) -> str:
        chatbot = self.get_chatbot()
        try:
            plugins.tts.args["voice"] = voice
            config = GenerationConfig(audio_output_path=audio_output_path)
            result, link = chatbot.chat_stream(query=prompt, config=config)
            def audio_file_generate(result):
                for path in result:
                    with open(path,mode="rb") as file:
                        bytes = file.read()
                        data = base64.b64encode(bytes)
                    yield f"data: {data}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(audio_file_generate(result), media_type="text/event-stream")
        except Exception as e:
            raise Exception(e)

    async def handle_create_speaker_embedding(self, spk_id):
        chatbot = self.get_chatbot()
        try:
            spk_embedding = chatbot.tts.create_speaker_embedding(spk_id)
            torch.save(spk_embedding, f'speaker_embeddings/spk_embed_{spk_id}.pt')
        except Exception as e:
            logger.info(f"create spk embedding failes! {e}")
            return {"create_spk": "fail"}
        return {"create_spk": "success"}


router = VoiceChatAPIRouter()

# We deliver an audio stream to the frontend which necessitating the use of SSE to manage real-time audio streaming.
# However, SSE isn't suitable for transmitting audio files directly.
# As a solution, we should split the API into /v1/voicechat/asr and /v1/voicechat/llm_tts.

@router.post("/v1/talkingbot/asr")
async def handle_talkingbot_asr(file: UploadFile = File(...)):
    file_name = file.filename
    logger.info(f'Received file: {file_name}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_audio_bytes")
    audio = audio.set_frame_rate(16000)
    # bytes to wav
    file_name = file_name +'.wav'
    audio.export(f"{file_name}", format="wav")
    asr_result = router.handle_voice_asr_request(file_name)
    return {"asr_result": asr_result}


@router.post("/v1/talkingbot/llm_tts")
async def talkingbot(request: Request):
    data = await request.json()
    text = data["text"]
    voice = data["voice"]
    knowledge_id = data["knowledge_id"]
    audio_output_path = data["audio_output_path"] if "audio_output_path" in data else "output_audio"

    logger.info(f'Received prompt: {text}, and use voice: {voice} knowledge_id: {knowledge_id}')

    return await router.handle_voice_chat_request(text, voice, audio_output_path)

@router.post("/v1/talkingbot/create_embedding")
async def create_speaker_embedding(file: UploadFile = File(...)):
    print(dir(file))
    file_name = file.filename
    # generate a unique id
    import uuid
    spk_id = f"spk_{uuid.uuid1()}"
    with open(f"tmp_spk_{file_name}", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file(f"tmp_spk_{file_name}")
    audio.export(f"{spk_id}", format="mp3")

    await router.handle_create_speaker_embedding(spk_id)
    return {"spk_id": spk_id}
