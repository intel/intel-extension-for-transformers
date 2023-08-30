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

import os

# Get the host and port from the environment variables
host = os.environ.get('MY_HOST')
port = os.environ.get('MY_PORT')

# Check if the environment variables are set and not empty
if host and port:
    # Combine the host and port to form the full URL
    HOST = f"http://{host}:{port}"
    API_COMPLETION = '/v1/completions'
    API_CHAT_COMPLETION = '/v1/chat/completions'
    API_AUDIO = '/v1/voicechat/completions'
    API_FINETUNE = '/v1/finetune'
    API_TEXT2IMAGE = '/v1/text2image'

    print("HOST URL:", HOST)
    print("Completions Endpoint:", API_COMPLETION)
    print("Chat completions Endpoint:", API_CHAT_COMPLETION)
    print("Voice cbat Endpoint:", API_AUDIO)
    print("Finerune Endpoint:", API_FINETUNE)
    print("Text to image Endpoint:", API_TEXT2IMAGE)
else:
    raise("Please set the environment variables MY_HOST and MY_PORT.")