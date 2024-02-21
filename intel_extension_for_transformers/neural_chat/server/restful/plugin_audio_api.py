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
from typing import Optional, List
from ...cli.log import logger
from fastapi import File, UploadFile
from ...plugins import plugins, get_plugin_instance
import base64
import torch

class AudioPluginAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def handle_voice_asr_request(self, filename: str, language: str = "auto") -> str:
        asr = get_plugin_instance("asr")
        asr.language = language
        try:
            return asr.audio2text(filename)
        except Exception as e:
            raise Exception(e)

    async def handle_voice_tts_request(self,
                                       text: str,
                                       voice: str,
                                       audio_output_path: Optional[str] = None,
                                       speedup: float = 1.0) -> str:

        plugins.tts.args['voice'] = voice
        plugins.tts.args['output_audio_path'] = audio_output_path
        tts = get_plugin_instance("tts")
        tts.speedup = speedup
        try:
            result = tts.post_llm_inference_actions(text)
            def audio_file_generate(result):
                if isinstance(result, List):
                    for path in result:
                        with open(path,mode="rb") as file:
                            bytes = file.read()
                            data = base64.b64encode(bytes)
                        yield f"data: {data}\n\n"
                else:
                    with open(result,mode="rb") as file:
                        bytes = file.read()
                        data = base64.b64encode(bytes)
                    yield f"data: {data}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(audio_file_generate(result), media_type="text/event-stream")
        except Exception as e:
            raise Exception(e)

    async def handle_create_speaker_embedding(self, spk_id):
        tts = get_plugin_instance("tts")
        try:
            spk_embedding = tts.create_speaker_embedding(spk_id)
            torch.save(spk_embedding, f'../../assets/speaker_embeddings/spk_embed_{spk_id}.pt')
            logger.info(f"create spk embedding succeed! {spk_id}")
        except Exception as e:
            logger.info(f"create spk embedding fails! {e}")
            return {"create_spk": "fail"}
        return {"create_spk": "success"}


router = AudioPluginAPIRouter()


@router.post("/plugin/audio/asr")
async def handle_talkingbot_asr(file: UploadFile = File(...), language: str = "auto"):
    file_name = file.filename
    logger.info(f'Received file: {file_name}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    from pydub import AudioSegment
    audio = AudioSegment.from_file("tmp_audio_bytes")
    audio = audio.set_frame_rate(16000)
    # bytes to wav
    file_name = file_name +'.wav'
    audio.export(f"{file_name}", format="wav")
    asr_result = router.handle_voice_asr_request(file_name, language=language)
    return {"asr_result": asr_result}


@router.post("/plugin/audio/tts")
async def talkingbot(request: Request):
    data = await request.json()
    text = data["text"]
    voice = data["voice"]
    speedup = float(data["speed"]) if "speed" in data else 1.0
    audio_output_path = data["audio_output_path"] if "audio_output_path" in data else "output_audio.wav"

    logger.info(f'Received prompt: {text}, and use voice: {voice}')

    return await router.handle_voice_tts_request(text, voice, audio_output_path, speedup)


@router.post("/plugin/audio/create_embedding")
async def create_speaker_embedding(file: UploadFile = File(...)):
    print(dir(file))
    file_name = file.filename
    # generate a unique id
    import uuid
    spk_id = f"spk_{str(uuid.uuid1())[:8]}"
    with open(f"tmp_spk_{file_name}", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    from pydub import AudioSegment
    audio = AudioSegment.from_file(f"tmp_spk_{file_name}")
    audio.export(f"{spk_id}", format="mp3")

    await router.handle_create_speaker_embedding(spk_id)
    return {"spk_id": spk_id}
