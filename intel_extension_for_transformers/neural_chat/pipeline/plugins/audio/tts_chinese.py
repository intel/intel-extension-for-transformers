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
    def __init__(self, stream_mode=False, output_audio_path="./response.wav", ):
        self.tts_executor = TTSExecutor()
        self.output_audio_path = output_audio_path
        self.stream_mode = stream_mode

    def text2speech(self, text, output_audio_path):
        "Chinese text to speech and dump to the output_audio_path."
        self.tts_executor(
            text=text,
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

    def stream_text2speech(self, generator, answer_speech_path):
        """Stream the generation of audios with an LLM text generator."""
        for idx, response in enumerate(generator):
            yield self.text2speech(response, f"{answer_speech_path}_{idx}.wav")

    def post_llm_inference_actions(self, text_or_generator):
        if self.stream_mode:
            def cache_words_into_sentences():
                buffered_texts = []
                hitted_ends = ['。', '！', '？', '；', '.', '!', '?', ';']
                prefix = ""
                # The stream_chat of glm is differnt with mpt, and its response is the full sentence, not single next token
                for response in text_or_generator:
                    if len(response.strip()) == 0:
                        continue
                    next_token = response.replace(prefix, "")
                    print(next_token, end="")
                    prefix = response
                    buffered_texts.append(next_token)
                    if(len(buffered_texts) > 5):
                        if next_token.strip() != "" and next_token.strip()[-1] in hitted_ends:
                            yield ''.join(buffered_texts)
                            buffered_texts = []
                if len(buffered_texts) > 0 and any([t.strip()!="" for t in buffered_texts]):
                    yield ''.join(buffered_texts)
            return self.stream_text2speech(cache_words_into_sentences(), self.output_audio_path)
        else:
            return self.text2speech(text_or_generator, self.output_audio_path)
