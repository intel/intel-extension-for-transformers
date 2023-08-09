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


class ChineseTextToSpeech:  # pragma: no cover
    def __init__(self, server_ip, port=443, protocol="http"):
        self.server_ip = server_ip
        self.port = port
        self.protocol = protocol
        self.executor = TTSOnlineClientExecutor()

    def text2speech(self, text, output_audio_path, spk_id=0):
        """Chinese text to speech and dump to the output_audio_path."""
        self.executor(input=text, server_ip=self.server_ip, port=self.port, protocol=self.protocol,
                    spk_id=spk_id, output=output_audio_path, play=False)
        return output_audio_path

    def stream_text2speech(self, generator, answer_speech_path, voice=0):
        """Stream the generation of audios with an LLM text generator."""
        for idx, response in enumerate(generator):
            yield self.text2speech(response, f"{answer_speech_path}_{idx}.wav", spk_id=voice)