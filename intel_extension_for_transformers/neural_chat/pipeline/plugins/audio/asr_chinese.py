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

from paddlespeech.cli.asr.infer import ASRExecutor
import time

class ChineseAudioSpeechRecognition():
    """Convert audio to text in Chinese."""
    def __init__(self):
        self.asr = ASRExecutor()

    def audio2text(self, audio_path):
        """Convert audio to text in Chinese.

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        start = time.time()
        result = self.asr(audio_file=audio_path)
        print(f"generated text in {time.time() - start} seconds, and the result is: {result}")
        return result

    def pre_llm_inference_actions(self, audio_path):
        return self.audio2text(audio_path)
