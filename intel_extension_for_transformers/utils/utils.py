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

def get_gpu_family():
    ''' Get gpu device family info.

    Return 'flex'|'max'|'arc' or assert

    Note, this function need to import intel_extension_for_pytorch

    Addtional info (common gpu name):
      'Intel(R) Data Center GPU Flex 170'
      'Intel(R) Data Center GPU Max 1100'
      'Intel(R) Arc(TM) A770 Graphics'
    '''

    import intel_extension_for_pytorch as ipex

    name = torch.xpu.get_device_name()
    if 'GPU Flex' in name:
        return 'flex'
    if 'GPU Max' in name:
        return 'max'
    if 'Arc(TM)' in name:
        return 'arc'
    assert False, "Unsupport GPU device: {}".format(name)