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
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import APIRouter
from typing import Optional
from ...cli.log import logger
from fastapi import File, UploadFile, Form
from ...config import GenerationConfig
import base64
import torch
from typing import Optional
from fastapi import Query


class FaceAnimationAPIRouter(APIRouter): # pragma: no cover

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

    async def handle_face_animation(self, image_path, audio_path, text, mode, voice):
        chatbot = self.get_chatbot()
        if mode == "fast":
            chatbot.face_animation.enhancer = ""
        elif mode == "quality":
            chatbot.face_animation.enhancer = "gfpgan"
        else:
            raise Exception(f"Unsupported mode: {mode}")
        try:
            video_path = chatbot.face_animate(image_path, audio_path, text, voice)
        except:
            raise Exception("Exception occurred when generating image from text.")
        else:
            logger.info(f'Face animation finished. Generated video path: {video_path}')
            return FileResponse(video_path)


router = FaceAnimationAPIRouter()

@router.post("/v1/talkingbot/face_animation")
async def handle_talkingbot_face_animation(image: UploadFile = File(...),
                                           audio: Optional[UploadFile] = None,
                                           text: Optional[str] = Form(None),
                                           mode: Optional[str] = Form("fast"),
                                           voice: Optional[str] = Form(None)): # pragma: no cover
    audio_file_name = audio.filename if audio else ""
    image_file_name = image.filename
    logger.info(f'Received audio: {audio_file_name}')
    logger.info(f'Received image: {image_file_name}')
    logger.info(f'Use mode: {mode}')
    if text is None and audio_file_name == "":
        raise Exception("The driven audio and input text should not be both None!")
    # write to image file
    with open(f"tmp_image.jpg", 'wb') as fout:
        content = await image.read()
        fout.write(content)
    if audio_file_name != "":
        with open("tmp_audio_bytes", 'wb') as fout:
            content = await audio.read()
            fout.write(content)
        from pydub import AudioSegment
        audio = AudioSegment.from_file("tmp_audio_bytes")
        audio = audio.set_frame_rate(16000)
        # bytes to wav
        audio.export(f"tmp_audio.wav", format="wav")

        response = await router.handle_face_animation(image_path=f"tmp_image.jpg",
                                                audio_path=f"tmp_audio.wav",
                                                text=None,
                                                mode=mode,
                                                voice=None)
    else:
        response = await router.handle_face_animation(image_path=f"tmp_image.jpg",
                                                audio_path=None,
                                                text=text,
                                                mode=mode,
                                                voice=voice)
    return response
