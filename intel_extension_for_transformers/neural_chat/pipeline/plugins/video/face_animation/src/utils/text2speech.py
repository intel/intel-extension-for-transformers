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

import os
import tempfile
from TTS.api import TTS


class TTSTalker:
    def __init__(self) -> None:
        model_name = TTS.list_models()[0]
        self.tts = TTS(model_name)

    def test(self, text, language="en"):
        tempf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=("." + "wav"),
        )

        self.tts.tts_to_file(text, speaker=self.tts.speakers[0], language=language, file_path=tempf.name)

        return tempf.name
