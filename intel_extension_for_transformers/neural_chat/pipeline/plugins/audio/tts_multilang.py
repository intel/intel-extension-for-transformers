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

from transformers import AutoModel
from huggingface_hub import hf_hub_download
import torch
from .models.bert_vits2.vits_model import SynthesizerTrn
from .models.bert_vits2.tools.sentence import split_by_language
from .models.bert_vits2.text.cleaner import clean_text, cleaned_text_to_sequence
from .models.bert_vits2.commons import intersperse
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
)

import numpy as np
import soundfile as sf
import time
import contextlib
import logging
import json

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO,
)


class BertVITSModel:
    def __init__(self, device="cpu", precision="bf16"):
        """Init the Bert and VITS models.

        Args:
            device: which device to use, should be cpu/cuda
            prevision: which precision to load with, should be fp32/bf16/int8 for cpu, fp32 for cuda
        """
        self.device = device
        self.precision = precision
        # pre-load the models
        if self.precision in ["fp32", "bf16"]:
            self.cn_bert_model = AutoModelForMaskedLM.from_pretrained(
                "hfl/chinese-roberta-wwm-ext-large"
            ).to(device)
        elif self.precision in ["int8"]:
            logging.info("Notice: VITS int8 is still experimental, please be careful to use!")
            from neural_compressor.utils.pytorch import load
            from transformers import BertForMaskedLM, AutoConfig

            # load the model without weights
            bert_cn = BertForMaskedLM(
                config=AutoConfig.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
            )
            ckpt_path = hf_hub_download(
                repo_id="spycsh/chinese-roberta-wwm-ext-large-int8",
                filename="bert_cn.pt",
            )
            self.cn_bert_model = load(ckpt_path, bert_cn)
        else:
            raise Exception("Unsupported precision, should be fp32/bf16/int8!")

        self.cn_tokenizer = AutoTokenizer.from_pretrained(
            "hfl/chinese-roberta-wwm-ext-large"
        )
        self.en_bert_model = DebertaV2Model.from_pretrained(
            "microsoft/deberta-v3-large"
        ).to(device)
        self.en_tokenizer = DebertaV2Tokenizer.from_pretrained(
            "microsoft/deberta-v3-large"
        )
        self.jp_bert_model = AutoModelForMaskedLM.from_pretrained(
            "ku-nlp/deberta-v2-large-japanese-char-wwm"
        ).to(device)
        self.jp_tokenizer = AutoTokenizer.from_pretrained(
            "ku-nlp/deberta-v2-large-japanese-char-wwm"
        )
        self.vits = SynthesizerTrn(
            n_vocab=112,  # len(symbols)
            use_spk_conditioned_encoder=True,
            use_noise_scaled_mas=True,
            use_mel_posterior_encoder=False,
            use_duration_discriminator=True,
            n_layers_q=3,
            use_spectral_norm=False,
        ).to(device)
        self.vits.eval()
        self.sdp_ratio = 0.2
        self.noise_scale = 0.6
        self.noise_scale_w = 0.8
        self.length_scale = 1

        if self.precision in ["fp32", "bf16"]:
            ckpt_path = hf_hub_download(
                repo_id="spycsh/bert-vits-thchs-6-8000",
                filename="G_8000.pth",
            )
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            iteration = ckpt_dict["iteration"]
            self.vits.load_state_dict(ckpt_dict["model"], strict=False)
            logging.info(f"VITS checkpoint loaded (iter: {iteration})")
        elif self.precision in ["int8"]:
            ckpt_path = hf_hub_download(
                repo_id="spycsh/bert-vits-thchs-6-8000",
                filename="G_8000_int8.pt",
            )
            from neural_compressor.utils.pytorch import load

            self.vits = load(ckpt_path, self.vits)
            logging.info(f"VITS int8 checkpoint loaded.")
        else:
            raise Exception("Unsupported precision, should be fp32/bf16/int8!")

        spk_id_map = hf_hub_download(
            repo_id="spycsh/bert-vits-thchs-6-8000",
            filename="spk_id_map.json",
        )
        with open(spk_id_map, 'r') as f:
            self.spk_id_map = json.load(f)

    def tts_fn(self, text, sid=0):
        sentences_list = split_by_language(text, target_languages=["zh", "ja", "en"])
        split_by_language_S = time.time()
        logging.info(
            f"**** split_by_language takes: {time.time() - split_by_language_S} sec"
        )
        text_to_generate = [[i[0] for i in sentences_list]]
        lang_to_generate = [[i[1] for i in sentences_list]]

        audio_list = []
        for idx, piece in enumerate(text_to_generate):
            audio = self.infer_multilang(piece, language=lang_to_generate[idx], sid=sid)
            audio_list.append(audio)

        audio_concat = np.concatenate(audio_list)
        return audio_concat

    def get_bert_feature(self, norm_text, word2ph, language_str):
        bert_models_map = {
            "ZH": self.cn_bert_model,
            "EN": self.en_bert_model,
            "JP": self.jp_bert_model,
        }
        tokenizer_map = {
            "ZH": self.cn_tokenizer,
            "EN": self.en_tokenizer,
            "JP": self.jp_tokenizer,
        }
        with torch.no_grad():
            inputs = tokenizer_map[language_str](norm_text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            S = time.time()
            if self.device == "cpu":
                if language_str != "ZH":
                    # currently en/jp DevertaV2Model only support bf16 inference
                    with torch.cpu.amp.autocast(
                        enabled=True, dtype=torch.bfloat16, cache_enabled=True
                    ) if self.precision in [
                        "int8",
                        "bf16",
                    ] else contextlib.nullcontext():
                        res = bert_models_map[language_str](
                            **inputs, output_hidden_states=True
                        )
                else:
                    with torch.cpu.amp.autocast(
                        enabled=True, dtype=torch.bfloat16, cache_enabled=True
                    ) if self.precision in ["bf16"] else contextlib.nullcontext():
                        res = bert_models_map[language_str](
                            **inputs, output_hidden_states=True
                        )
            else:
                res = bert_models_map[language_str](**inputs, output_hidden_states=True)
            logging.info(f"**** {language_str} bert infer time: {time.time() - S} sec")
            res = res["hidden_states"][-3][0].cpu()
        word2phone = word2ph
        phone_level_feature = []
        for i in range(len(word2phone)):
            repeat_feature = res[i].repeat(word2phone[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        bert_feature = phone_level_feature.T

        return bert_feature

    def get_text(self, text, language_str):
        norm_text, phone, tone, word2ph = clean_text(text, language_str)
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

        # default add blank
        phone = intersperse(phone, 0)
        tone = intersperse(tone, 0)
        language = intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
        # get bert embedding for the current text
        bert_ori = self.get_bert_feature(
            norm_text, word2ph, language_str
        )  # (1024, len(phone))
        del word2ph
        assert bert_ori.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "JP":
            bert = torch.zeros(1024, len(phone))
            ja_bert = bert_ori
            en_bert = torch.zeros(1024, len(phone))
        elif language_str == "EN":
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(1024, len(phone))
            en_bert = bert_ori
        else:
            raise ValueError("language_str should be ZH, JP or EN")

        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, en_bert, phone, tone, language

    def infer_multilang(self, text, language, sid):
        bert, ja_bert, en_bert, phones, tones, lang_ids = [], [], [], [], [], []
        emo = torch.Tensor([0])  # disable emotion for now

        skip_start = False
        skip_end = False
        for idx, (txt, lang) in enumerate(zip(text, language)):
            skip_start = (idx != 0) or (skip_start and idx == 0)
            skip_end = (idx != len(text) - 1) or (skip_end and idx == len(text) - 1)
            lang = lang.upper()
            lang = "JP" if lang == "JA" else lang

            ## jieba will load model for the first instance
            get_text_S = time.time()
            (
                temp_bert,
                temp_ja_bert,
                temp_en_bert,
                temp_phones,
                temp_tones,
                temp_lang_ids,
            ) = self.get_text(txt, lang)
            print(f"**** get_text takes: {time.time() - get_text_S} sec")
            if skip_start:
                temp_bert = temp_bert[:, 1:]
                temp_ja_bert = temp_ja_bert[:, 1:]
                temp_en_bert = temp_en_bert[:, 1:]
                temp_phones = temp_phones[1:]
                temp_tones = temp_tones[1:]
                temp_lang_ids = temp_lang_ids[1:]
            if skip_end:
                temp_bert = temp_bert[:, :-1]
                temp_ja_bert = temp_ja_bert[:, :-1]
                temp_en_bert = temp_en_bert[:, :-1]
                temp_phones = temp_phones[:-1]
                temp_tones = temp_tones[:-1]
                temp_lang_ids = temp_lang_ids[:-1]
            bert.append(temp_bert)
            ja_bert.append(temp_ja_bert)
            en_bert.append(temp_en_bert)
            phones.append(temp_phones)
            tones.append(temp_tones)
            lang_ids.append(temp_lang_ids)
        bert = torch.concatenate(bert, dim=1)
        ja_bert = torch.concatenate(ja_bert, dim=1)
        en_bert = torch.concatenate(en_bert, dim=1)
        phones = torch.concatenate(phones, dim=0)
        tones = torch.concatenate(tones, dim=0)
        lang_ids = torch.concatenate(lang_ids, dim=0)
        with torch.no_grad():
            x_tst = phones.to(self.device).unsqueeze(0)
            tones = tones.to(self.device).unsqueeze(0)
            lang_ids = lang_ids.to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            ja_bert = ja_bert.to(self.device).unsqueeze(0)
            en_bert = en_bert.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
            del phones
            speakers = torch.LongTensor([sid]).to(self.device)
            infer_S = time.time()
            audio = (
                self.vits.infer(  # net_g.infer
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    en_bert,
                    emo=emo,  # disable emotion for now
                    sdp_ratio=self.sdp_ratio,
                    noise_scale=self.noise_scale,
                    noise_scale_w=self.noise_scale_w,
                    length_scale=self.length_scale,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        logging.info(f"**** net_g.infer takes: {time.time() - infer_S} sec")
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers, ja_bert, en_bert, emo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio


class MultilangTextToSpeech:
    def __init__(self, output_audio_path="./response.wav", voice="default", device="cpu", precision="bf16"):
        self.output_audio_path = output_audio_path
        self.voice = voice
        self.bert_vits_model = BertVITSModel(device, precision)

    def text2speech(self, text, output_audio_path, voice="default"):
        "Multilingual text to speech and dump to the output_audio_path."
        logging.info(text)
        all_speech = self.bert_vits_model.tts_fn(text, sid=self.bert_vits_model.spk_id_map[voice])
        sf.write(output_audio_path, all_speech, samplerate=44100)
        return output_audio_path

    def post_llm_inference_actions(self, text):
        from intel_extension_for_transformers.neural_chat.plugins import plugins
        self.voice = plugins.tts_multilang.args["voice"] \
            if plugins.tts_multilang.args['voice'] in self.bert_vits_model.spk_id_map else "default"
        self.output_audio_path = plugins.tts_multilang.args['output_audio_path'] \
            if plugins.tts_multilang.args['output_audio_path'] else "./response.wav"
        return self.text2speech(text, self.output_audio_path, self.voice)
