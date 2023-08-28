# !/usr/bin/env python
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

from locust import HttpUser, task, between
from intel_extension_for_transformers.neural_chat.tests.restful.config import API_COMPLETION, API_CHAT_COMPLETION, API_ASR, API_TTS, API_FINETUNE, API_TEXT2IMAGE
import time
from intel_extension_for_transformers.neural_chat.server.restful.openai_protocol import CompletionRequest, ChatCompletionRequest
from datasets import Dataset, Audio


# locust will create a FeedbackUser instance for each user
class FeedbackUser(HttpUser):
    # each simulated user will wait 1~2 seconds for the next operation
    wait_time = between(0.5, 2)

    @task
    def test_completions(self):
        time.sleep(0.01)
        request = CompletionRequest(
            model="mpt-7b-chat",
            prompt="This is a test."
        )
        self.client.post(API_COMPLETION, data=request)

    @task
    def test_chat_completions(self):
        time.sleep(0.01)
        request = ChatCompletionRequest(
            model="mpt-7b-chat",
            messages=[
                {"role": "system","content": "You are a helpful assistant."},
                {"role": "user","content": "Hello!"}
            ]
        )
        self.client.post(API_CHAT_COMPLETION, data=request)

    @task
    def test_asr(self):
        audio_path = "../../../assets/audio/pat.wav"
        audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        self.client.post(API_ASR, data=waveform)

    @task
    def test_tts(self):
        text = "Welcome to Neural Chat"
        self.client.post(API_TTS, data=text)

    @task
    def test_text2image(self):
        text = "A running horse."
        self.client.post(API_TEXT2IMAGE, data=text)

    @task
    def test_finetune(self):
        self.client.post(API_FINETUNE)
        time.sleep(2)