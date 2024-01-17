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

import paddle
from paddlespeech.cli.tts import TTSExecutor

class ChineseTextToSpeech():
    def __init__(self):
        self.tts_executor = TTSExecutor()

    def text2speech(self, input, output_audio_path): # pragma: no cover
        "Chinese text to speech and dump to the output_audio_path."
        self.tts_executor(
            text=input,
            output=output_audio_path,
            am='fastspeech2_csmsc',
            am_config=None,
            am_ckpt=None,
            am_stat=None,
            spk_id=0,
            phones_dict=None,
            tones_dict=None,
            speaker_dict=None,
            voc='pwgan_csmsc',
            voc_config=None,
            voc_ckpt=None,
            voc_stat=None,
            lang='zh',
            device=paddle.get_device())
        return output_audio_path

    def post_llm_inference_actions(self, text, output_audio_path): # pragma: no cover
        return self.text2speech(text, output_audio_path)
