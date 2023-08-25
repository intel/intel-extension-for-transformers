#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

"""Utility for data augmentation."""

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from enum import Enum


EOS = '</s>'


class AugmenterType(Enum):
    """Enumeration of types of augmentation."""
    TEXTGENERATIONAUG = "textgenerationaug"
    KEYBOARDAUG = "KeyboardAug"
    OCRAUG = "OcrAug"
    SPELLINGAUG = "SpellingAug"
    CONTEXTUALWORDEMBSFORSENTENCEAUG = "ContextualWordEmbsForSentenceAug"


AUGMENTER_MAPPING = {
    AugmenterType.KEYBOARDAUG.value: nac,
    AugmenterType.OCRAUG.value: nac,
    AugmenterType.SPELLINGAUG.value: naw,
    AugmenterType.CONTEXTUALWORDEMBSFORSENTENCEAUG.value: nas,

}


def get_augmenter_from_type(aug_type: str):
    """Get nlpaug's augmenter by augment_type name.

    The nlpaug is a library helps you with augmenting nlp for your machine learning projects.
    It provide many augmenter, please refer to https://github.com/makcedward/nlpaug#augmenter.
    """
    assert aug_type in AUGMENTER_MAPPING, "Unspported the augmenter type:{}".format(aug_type)
    return AUGMENTER_MAPPING[aug_type]
