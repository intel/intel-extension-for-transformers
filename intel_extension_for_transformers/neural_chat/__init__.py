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

from .config import PipelineConfig
from .config import GenerationConfig
from .config import (
    TextGenerationFinetuningConfig,
    SummarizationFinetuningConfig,
    CodeGenerationFinetuningConfig,
    TTSFinetuningConfig
)
from .chatbot import build_chatbot
from .chatbot import finetune_model
from .chatbot import optimize_model
from .server.neuralchat_server import NeuralChatServerExecutor
from .server.neuralchat_client import TextChatClientExecutor, VoiceChatClientExecutor, FinetuningClientExecutor
from .plugins import plugins
