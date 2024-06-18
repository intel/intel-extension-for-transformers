# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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

import re, sys ,os
import cn2an
import opencc

converter = opencc.OpenCC('lexicon/zaonhe.json')

# List of (Latin alphabet, ipa) pairs:
_latin_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('A', 'ᴇ'),
    ('B', 'bi'),
    ('C', 'si'),
    ('D', 'di'),
    ('E', 'i'),
    ('F', 'ᴇf'),
    ('G', 'dʑi'),
    ('H', 'ᴇtɕʰ'),
    ('I', 'ᴀi'),
    ('J', 'dʑᴇ'),
    ('K', 'kʰᴇ'),
    ('L', 'ᴇl'),
    ('M', 'ᴇm'),
    ('N', 'ᴇn'),
    ('O', 'o'),
    ('P', 'pʰi'),
    ('Q', 'kʰiu'),
    ('R', 'ᴀl'),
    ('S', 'ᴇs'),
    ('T', 'tʰi'),
    ('U', 'ɦiu'),
    ('V', 'vi'),
    ('W', 'dᴀbɤliu'),
    ('X', 'ᴇks'),
    ('Y', 'uᴀi'),
    ('Z', 'zᴇ')
]]

def _number_to_shanghainese(num):
    num = cn2an.an2cn(num).replace('一十','十').replace('二十', '廿').replace('二', '两')
    return re.sub(r'((?:^|[^三四五六七八九])十|廿)两', r'\1二', num)

def number_to_shanghainese(text):
    return re.sub(r'\d+(?:\.?\d+)?', lambda x: _number_to_shanghainese(x.group()), text)

def latin_to_ipa(text):
    for regex, replacement in _latin_to_ipa:
        text = re.sub(regex, replacement, text)
    return text

def shanghainese_to_ipa(text):
    text = number_to_shanghainese(text.upper())
    text = converter.convert(text).replace('-','').replace('$',' ')
    text = re.sub(r'[A-Z]', lambda x: latin_to_ipa(x.group())+' ', text)
    text = re.sub(r'[、；：]', '，', text)
    text = re.sub(r'\s*，\s*', ', ', text)
    text = re.sub(r'\s*。\s*', '. ', text)
    text = re.sub(r'\s*？\s*', '? ', text)
    text = re.sub(r'\s*！\s*', '! ', text)
    text = re.sub(r'\s*$', '', text)
    return text

def shanghainese_cleaners(text):
    text = shanghainese_to_ipa(text)
    if re.match(r'[^\.,!\?\-…~]', text[-1]):
        text += '.'
    return text
