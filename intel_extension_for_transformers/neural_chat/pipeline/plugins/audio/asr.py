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

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import Audio, Dataset
import time
import contextlib
from pydub import AudioSegment
import numpy as np
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class AudioSpeechRecognition():
    """Convert audio to text."""
    def __init__(self, model_name_or_path="openai/whisper-small", bf16=False, language=None, device="cpu"):
        if device == "auto":
            device = get_device_type()
        self.device = device
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        self.model.eval()
        self.bf16 = bf16
        if self.bf16:
            import intel_extension_for_pytorch as ipex
            self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
        self.language = language

    def _audiosegment_to_librosawav(self, audiosegment):
        # https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
        # This way is faster than librosa.load or HuggingFace Dataset wrapper
        channel_sounds = audiosegment.split_to_mono()[:1]   # only select the first channel
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        fp_arr = fp_arr.reshape(-1)

        return fp_arr

    def _convert_audio_type(self, audio_path): # pragma: no cover
        logging.warning("[ASR WARNING] Recommend to use mp3 or wav input audio type!")
        audio_file_name = audio_path.split(".")[0]
        AudioSegment.from_file(audio_path).export(f"{audio_file_name}.mp3", format="mp3")
        return f"{audio_file_name}.mp3"

    def audio2text(self, audio_path):
        """Convert audio to text

        audio_path: the path to the input audio, e.g. ~/xxx.mp3
        """
        start = time.time()
        if audio_path.split(".")[-1] in ['flac', 'ogg', 'aac', 'm4a']:
            audio_path = self._convert_audio_type(audio_path)
        elif audio_path.split(".")[-1] not in ['mp3', 'wav']:
            raise Exception("[ASR ERROR] Audio format not supported!")

        try:
            waveform = AudioSegment.from_file(audio_path).set_frame_rate(16000)
            waveform = self._audiosegment_to_librosawav(waveform)
        except Exception as e:
            logging.error(f"[ASR] audiosegment to librosa wave fail: {e}")
            audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
            waveform = audio_dataset[0]["audio"]['array']

        # pylint: disable=E1101
        inputs = self.processor.feature_extractor(waveform, return_tensors="pt",
                        sampling_rate=16_000).input_features.to(self.device)
        with torch.cpu.amp.autocast() if self.bf16 else contextlib.nullcontext():
            if self.language is None:
                predicted_ids = self.model.generate(inputs)
            elif self.language == "auto":
                self.model.config.forced_decoder_ids = None
                predicted_ids = self.model.generate(inputs)
            else:
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language,
                                                                                task="transcribe")
                self.model.config.forced_decoder_ids = self.forced_decoder_ids
                predicted_ids = self.model.generate(inputs)
        # pylint: disable=E1101
        result = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True, normalize=True)[0]
        if self.language == "auto" or self.language == "cn":
            from zhconv import convert
            result = convert(result, 'zh-cn')
        logging.info(f"generated text in {time.time() - start} seconds, and the result is: {result}")
        return result

    def pre_llm_inference_actions(self, audio_path):
        return self.audio2text(audio_path)
