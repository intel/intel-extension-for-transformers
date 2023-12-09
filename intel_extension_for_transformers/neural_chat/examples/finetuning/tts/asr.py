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


from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.asr import AudioSpeechRecognition
import argparse
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

parser = argparse.ArgumentParser(
                    prog='asr',
                    description='Audio Speech Recognition')
parser.add_argument('-i', '--input_audio')
parser.add_argument('-m', '--model_name_or_path', default="openai/whisper-tiny")
parser.add_argument('-d', '--device', default="cuda")
args = parser.parse_args()
asr = AudioSpeechRecognition(model_name_or_path=args.model_name_or_path, device=args.device)
text = asr.audio2text(args.input_audio)
logging.info(text)
