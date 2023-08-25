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

from paddlespeech.server.bin.paddlespeech_client import TTSOnlineClientExecutor
from ....plugins import register_plugin

@register_plugin('tts_chinese')
class ChineseTextToSpeech():  # pragma: no cover
    def __init__(self, output_audio_path="./response.wav", spk_id=0,
                 stream_mode=False, server_ip="127.0.0.1", port=443, protocol="http", device="cpu"):
        self.server_ip = server_ip
        self.port = port
        self.protocol = protocol
        self.executor = TTSOnlineClientExecutor()
        self.stream_mode = stream_mode
        self.spk_id = spk_id
        self.output_audio_path = output_audio_path
        self.device = device

    def text2speech(self, text):
        """Chinese text to speech and dump to the output_audio_path."""
        self.executor(input=text, server_ip=self.server_ip, port=self.port, protocol=self.protocol,
                    spk_id=self.spk_id, output=self.output_audio_path, play=False)
        return self.output_audio_path

    def stream_text2speech(self, generator):
        """Stream the generation of audios with an LLM text generator."""
        for idx, response in enumerate(generator):
            yield self.text2speech(response, f"{self.output_audio_path}_{idx}.wav", spk_id=self.spk_id)

    def post_llm_inference_actions(self, text_or_generator):
        if self.stream_mode:
            return self.stream_text2speech(text_or_generator)
        else:
            return self.text2speech(text_or_generator)
