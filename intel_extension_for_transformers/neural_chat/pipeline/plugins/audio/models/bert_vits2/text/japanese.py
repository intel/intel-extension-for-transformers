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


# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from ..text.symbols import punctuation, symbols

from num2words import num2words

import pyopenjtalk
import jaconv


def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()
    if text == "ー":
        return ["ー"]
    elif text.startswith("ー"):
        return ["ー"] + kata2phoneme(text[1:])
    res = []
    while text:
        if re.match(_MARKS, text):
            res.append(text)
            text = text[1:]
            continue
        if text.startswith("ー"):
            text = text[1:]
            continue
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")
        break
    # res = _COLON_RX.sub(":", res)
    return res


def hira2kata(text: str) -> str:
    return jaconv.hira2kata(text)


_SYMBOL_TOKENS = set(list("・、。？！"))
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


def text2kata(text: str) -> str:
    parsed = pyopenjtalk.run_frontend(text)

    res = []
    for parts in parsed:
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        if yomi:
            if re.match(_MARKS, yomi):
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        else:
            if word in _SYMBOL_TOKENS:
                res.append(word)
            elif word in ("っ", "ッ"):
                res.append("ッ")
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
    return hira2kata("".join(res))


def text2sep_kata(text: str) -> (list, list):
    parsed = pyopenjtalk.run_frontend(text)

    res = []
    sep = []
    for parts in parsed:
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        if yomi:
            if re.match(_MARKS, yomi):
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        else:
            if word in _SYMBOL_TOKENS:
                res.append(word)
            elif word in ("っ", "ッ"):
                res.append("ッ")
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
        sep.append(word)
    return sep, [hira2kata(i) for i in res], get_accent(parsed)


def get_accent(parsed):
    labels = pyopenjtalk.make_label(parsed)

    phonemes = []
    accents = []
    for n, label in enumerate(labels):
        phoneme = re.search(r"\-([^\+]*)\+", label).group(1)
        if phoneme not in ["sil", "pau"]:
            phonemes.append(phoneme.replace("cl", "q").lower())
        else:
            continue
        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))
        if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:
            a2_next = -1
        else:
            a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
        # Falling
        if a1 == 0 and a2_next == a2 + 1:
            accents.append(-1)
        # Rising
        elif a2 == 1 and a2_next == 2:
            accents.append(1)
        else:
            accents.append(0)
    return list(zip(phonemes, accents))


_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",
    ">": "大なり",
    "@": "アット",
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    "f": "エフ",
    "g": "ジー",
    "h": "エイチ",
    "i": "アイ",
    "j": "ジェー",
    "k": "ケー",
    "l": "エル",
    "m": "エム",
    "n": "エヌ",
    "o": "オー",
    "p": "ピー",
    "q": "キュー",
    "r": "アール",
    "s": "エス",
    "t": "ティー",
    "u": "ユー",
    "v": "ブイ",
    "w": "ダブリュー",
    "x": "エックス",
    "y": "ワイ",
    "z": "ゼット",
    "α": "アルファ",
    "β": "ベータ",
    "γ": "ガンマ",
    "δ": "デルタ",
    "ε": "イプシロン",
    "ζ": "ゼータ",
    "η": "イータ",
    "θ": "シータ",
    "ι": "イオタ",
    "κ": "カッパ",
    "λ": "ラムダ",
    "μ": "ミュー",
    "ν": "ニュー",
    "ξ": "クサイ",
    "ο": "オミクロン",
    "π": "パイ",
    "ρ": "ロー",
    "σ": "シグマ",
    "τ": "タウ",
    "υ": "ウプシロン",
    "φ": "ファイ",
    "χ": "カイ",
    "ψ": "プサイ",
    "ω": "オメガ",
}


_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")


def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])


def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res


def is_japanese_character(char):
    japanese_ranges = [
        (0x3040, 0x309F),
        (0x30A0, 0x30FF),
        (0x4E00, 0x9FFF),
        (0x3400, 0x4DBF),
        (0x20000, 0x2A6DF),
    ]
    char_code = ord(char)
    for start, end in japanese_ranges:
        if start <= char_code <= end:
            return True

    return False


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "−": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    return replaced_text


def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    res = res.replace("゙", "")
    return res


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def handle_long(sep_phonemes):
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    return sep_phonemes


tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-large-japanese-char-wwm")


def align_tones(phones, tones):
    res = []
    for pho in phones:
        temp = [0] * len(pho)
        for idx, p in enumerate(pho):
            if len(tones) == 0:
                break
            if p == tones[0][0]:
                temp[idx] = tones[0][1]
                if idx > 0:
                    temp[idx] += temp[idx - 1]
                tones.pop(0)
        temp = [0] + temp
        temp = temp[:-1]
        if -1 in temp:
            temp = [i + 1 for i in temp]
        res.append(temp)
    res = [i for j in res for i in j]
    assert not any([i < 0 for i in res]) and not any([i > 1 for i in res])
    return res


def rearrange_tones(tones, phones):
    res = [0] * len(tones)
    for i in range(len(tones)):
        if i == 0:
            if tones[i] not in punctuation:
                res[i] = 1
        elif tones[i] == prev:
            if phones[i] in punctuation:
                res[i] = 0
            else:
                res[i] = 1
        elif tones[i] > prev:
            res[i] = 2
        elif tones[i] < prev:
            res[i - 1] = 3
            res[i] = 1
        prev = tones[i]
    return res


def g2p(norm_text):
    sep_text, sep_kata, acc = text2sep_kata(norm_text)
    sep_tokenized = []
    for i in sep_text:
        if i not in punctuation:
            sep_tokenized.append(tokenizer.tokenize(i))
        else:
            sep_tokenized.append([i])

    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    for i in sep_phonemes:
        for j in i:
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)
    tones = align_tones(sep_phonemes, acc)

    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)

        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    # tones = [0] + rearrange_tones(tones, phones[1:-1]) + [0]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones)
    return phones, tones, word2ph
