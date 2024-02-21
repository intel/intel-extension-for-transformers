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

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, set_seed
from datasets import load_dataset, Audio, Dataset, Features, ClassLabel
import os
import torch
from typing import Any, Dict, List, Union
from transformers import SpeechT5HifiGan
import soundfile as sf
import numpy as np
import contextlib
from pydub import AudioSegment

import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

from .utils.english_normalizer import EnglishNormalizer
from .utils.reduce_noise import NoiseReducer
from intel_extension_for_transformers.neural_chat.utils.common import get_device_type

class TextToSpeech():
    """Convert text to speech with a driven speaker embedding

    1) Default voice (Original model + Proved good default speaker embedding from trained dataset)
    2) Finetuned voice (Fine-tuned offline model of specific person's voice + corresponding embedding)
    3) Customized voice (Original model + User's customized input voice embedding)
    """
    def __init__(self, output_audio_path="./response.wav", voice="default", stream_mode=False, device="cpu",
                 reduce_noise=False, speedup=1.0):
        """Make sure your export LD_PRELOAD=<path to libiomp5.so and libtcmalloc> beforehand."""
        # default setting
        if device == "auto":
            device = get_device_type()
        self.device = device
        self.original_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.voice = voice
        self.output_audio_path = output_audio_path
        self.stream_mode = stream_mode
        self.spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
        try:
            from speechbrain.pretrained import EncoderClassifier
            self.speaker_model = EncoderClassifier.from_hparams(
                source=self.spk_model_name,
                run_opts={"device": "cpu"},
                savedir=os.path.join("/tmp", self.spk_model_name))
        except Exception as e: # pragma: no cover
            logging.warning("[TTS Warning] speaker model fail to load, so speaker embedding creating is disabled.")
            self.speaker_model = None
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        self.vocoder.eval()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_speaker_embedding = None
        if os.path.exists(os.path.join(script_dir, '../../../assets/speaker_embeddings/spk_embed_default.pt')):
            default_speaker_embedding_path = os.path.join(
                script_dir, '../../../assets/speaker_embeddings/spk_embed_default.pt')
            self.default_speaker_embedding = torch.load(default_speaker_embedding_path)
        elif os.path.exists('spk_embed_default.pt'):    # for notebook
            self.default_speaker_embedding = torch.load('spk_embed_default.pt')
        else: # pragma: no cover
            import subprocess
            try:
                p = subprocess.Popen(["wget",
                                "https://github.com/intel/intel-extension-for-transformers/raw/main/"
                                "intel_extension_for_transformers/neural_chat/assets/speaker_embeddings/"
                                "spk_embed_default.pt"])
                p.wait()
                self.default_speaker_embedding = torch.load('spk_embed_default.pt')
            except Exception as e:
                logging.warning("Warning! Need to prepare speaker_embeddings, will use the backup embedding.")
                self.default_speaker_embedding = torch.zeros((1, 512))

        # preload the demo model in case of time-consuming runtime loading
        self.demo_model = None
        if os.path.exists("demo_model.pt"):  # pragma: no cover
            self.demo_model = torch.load("demo_model.pt", map_location=device)

        self.male_speaker_embeddings = None
        pat_speaker_embedding_path = os.path.join(script_dir, '../../../assets/speaker_embeddings/spk_embed_male.pt')
        if os.path.exists(pat_speaker_embedding_path):
            self.male_speaker_embeddings = torch.load(pat_speaker_embedding_path)

        self.normalizer = EnglishNormalizer()
        self.noise_reducer = NoiseReducer() if reduce_noise else None
        self.speedup = speedup

    def _audiosegment_to_librosawav(self, audiosegment):
        # https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
        # This way is faster than librosa.load or HuggingFace Dataset wrapper
        channel_sounds = audiosegment.split_to_mono()[:1]   # only select the first channel
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max
        fp_arr = fp_arr.reshape(-1)

        return fp_arr

    def create_speaker_embedding(self, driven_audio_path):
        """Create the speaker's embedding.

        driven_audio_path: the driven audio of that speaker
        """
        if self.speaker_model is None:
            raise Exception("Unable to create a speaker embedding! Please check the speaker model.")
        try:
            waveform = AudioSegment.from_file(driven_audio_path).set_frame_rate(16000)
            waveform = self._audiosegment_to_librosawav(waveform)
        except Exception as e:
            logging.error(f"[TTS] audiosegment to librosa wave fail: {e}")
            audio_dataset = Dataset.from_dict({"audio":
                [driven_audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
            waveform = audio_dataset[0]["audio"]['array']
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(torch.tensor(waveform).to("cpu"))
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2) # [1,1,512]
            speaker_embeddings = speaker_embeddings[0] # [1,512]
        return speaker_embeddings.to(self.device)

    def _lookup_voice_embedding(self, voice):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(script_dir, f'../../../assets/speaker_embeddings/spk_embed_{voice}.pt')):
            specific_speaker_embedding_path = os.path.join(script_dir,
                                        f"../../../assets/speaker_embeddings/spk_embed_{voice}.pt")
            return torch.load(specific_speaker_embedding_path)
        elif os.path.exists(f'spk_embed_{voice}.pt'):    # for notebook
            return torch.load(f'spk_embed_{voice}.pt')
        else:
            logging.warning("No customized speaker embedding is found! Use the default one")
            return self.default_speaker_embedding

    def _batch_long_text(self, text, batch_length):
        """Batch the long text into sequences of shorter sentences."""
        res = []
        hitted_ends = ['.', '?', '!', '。', ";"]
        idx = 0
        cur_start = 0
        cur_end = -1
        while idx < len(text):
            if idx - cur_start > batch_length:
                if cur_end != -1 and cur_end > cur_start:
                    res.append(text[cur_start:cur_end+1])
                else:
                    logging.warning(
                        f"[TTS Warning] Check your input text and it should be split by one of {hitted_ends} "
                        + f"in each {batch_length} characters! Try to add batch_length!"
                    )
                    cur_end = cur_start+batch_length-1
                    res.append(text[cur_start:cur_end+1])
                idx = cur_end
                cur_start = cur_end + 1
            if text[idx] in hitted_ends:
                cur_end = idx
            idx += 1
        # deal with the last sequence
        res.append(text[cur_start:idx])
        res = [i + "." for i in res]    # avoid unexpected end of sequence
        return res

    def _speedup(self, path, speed):
        from pydub import AudioSegment
        from pydub.effects import speedup
        sound = AudioSegment.from_file(path)
        speedup(sound, playback_speed=speed).export(path)

    def text2speech(self, text, output_audio_path="./response.wav", voice="default", speedup=1.0,
                    do_batch_tts=False, batch_length=400):
        """Text to speech.

        text: the input text
        voice: default/male/female/...
        batch_length: the batch length for splitting long texts into batches to do text to speech
        """
        logging.info(text)
        if batch_length > 600 or batch_length < 50:
            raise Exception(f"[TTS] Invalid batch_length {batch_length}, should be between 50 and 600!")
        text = self.normalizer.correct_abbreviation(text)
        text = self.normalizer.correct_number(text)
        # Do the batching of long texts
        if len(text) > batch_length or do_batch_tts:
            texts = self._batch_long_text(text, batch_length)
        else:
            texts = [text]
        logging.info(f"[TTS] batched texts: {texts}")
        model = self.original_model
        speaker_embeddings = self.default_speaker_embedding
        if voice == "male":
            if self.demo_model == None:
                logging.warning("Finetuned model is not found! Use the default one")
            else: # pragma: no cover
                model = self.demo_model
            if self.male_speaker_embeddings == None: # pragma: no cover
                logging.warning("Male speaker embedding is not found! Use the default one")
            else:
                speaker_embeddings = self.male_speaker_embeddings
        elif voice != "default":
            speaker_embeddings = self._lookup_voice_embedding(voice)
        all_speech = np.array([])
        for text_in in texts:
            inputs = self.processor(text=text_in, return_tensors="pt")
            with torch.no_grad():
                spectrogram = model.generate_speech(
                    inputs["input_ids"].to(self.device), speaker_embeddings.to(self.device))
                speech = self.vocoder(spectrogram)
                all_speech = np.concatenate([all_speech, speech.cpu().numpy()])
                all_speech = np.concatenate([all_speech, np.array([0 for i in range(8000)])])  # pad after each end
        sf.write(output_audio_path, all_speech, samplerate=16000)
        if self.noise_reducer:
            output_audio_path = self.noise_reducer.reduce_audio_amplify(output_audio_path, all_speech)
        if speedup != 1.0:
            self._speedup(output_audio_path, speedup)
        return output_audio_path

    def stream_text2speech(self, generator, output_audio_path, voice="default", speedup=1.0):
        """Stream the generation of audios with an LLM text generator."""
        for idx, response in enumerate(generator):
            yield self.text2speech(response, f"{output_audio_path}_{idx}.wav", voice, speedup)


    def post_llm_inference_actions(self, text_or_generator):
        from intel_extension_for_transformers.neural_chat.plugins import plugins
        self.voice = plugins.tts.args["voice"]
        self.output_audio_path = plugins.tts.args['output_audio_path'] \
            if plugins.tts.args['output_audio_path'] else "./response.wav"
        if self.stream_mode: # pragma: no cover
            def cache_words_into_sentences():
                buffered_texts = []
                hitted_ends = ['.', '!', '?', ';', ':']
                for new_text in text_or_generator:
                    logging.info(f"new text: ==={new_text}===")
                    if len(new_text.strip()) == 0:
                        continue
                    buffered_texts.append(new_text)
                    if(len(buffered_texts) > 5):
                        if new_text.endswith('... ') or new_text.strip()[-1] in hitted_ends:
                            yield ''.join(buffered_texts)
                            buffered_texts = []
                # output the trailing sequence
                if len(buffered_texts) > 0:
                    yield ''.join(buffered_texts)
            return self.stream_text2speech(
                cache_words_into_sentences(), self.output_audio_path, self.voice, self.speedup)
        else:
            return self.text2speech(text_or_generator, self.output_audio_path, self.voice, self.speedup)
