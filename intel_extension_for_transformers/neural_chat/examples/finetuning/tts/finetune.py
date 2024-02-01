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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.finetuning.tts_finetuning import TTSFinetuning
from intel_extension_for_transformers.neural_chat.config import TTSFinetuningConfig, TTSDatasetArguments, TTSModelArguments
import torch
import os

workdir = os.getcwd()
data_args = TTSDatasetArguments(audio_folder_path=os.path.join(workdir, "audios"),
                                text_folder_path=os.path.join(workdir, "texts"),)
model_args = TTSModelArguments(step=1000, warmup_step=125, learning_rate=1e-5)
finetuning_config = TTSFinetuningConfig(data_args, model_args)

tts_fintuner = TTSFinetuning(finetuning_config=finetuning_config)
finetuned_model = tts_fintuner.finetune()

torch.save(finetuned_model, "finetuned_model.pt")
