#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import unittest
import sys
import numpy as np
import os
import shutil
from intel_extension_for_transformers.backends.neural_engine.compile.graph import Graph
from intel_extension_for_transformers.backends.neural_engine.compile.sub_graph.pattern import PATTERNS

file_name = os.path.splitext(os.path.basename(__file__))[0]


class TestTorchOP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        os.remove('conf.yaml')
        pass

    def test_1(self):
        text = '''
model:
  name: model
  operator:
    input_data:
      type: Input
      output:
        input_ids.1:
          dtype: s32
          shape: [-1, -1]
        attention_mask.1:
          dtype: s32
          shape: [-1, -1]
        x.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x0.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x1.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x2.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x3.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x4.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x5.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x6.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x7.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x8.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x9.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x10.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x11.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x12.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x13.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x14.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x15.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x16.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x17.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x18.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x19.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x20.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x21.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x22.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x23.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x24.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x25.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x26.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x27.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x28.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x29.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x30.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x31.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x32.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x33.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x34.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x35.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x36.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x37.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x38.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x39.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x40.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x41.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x42.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x43.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x44.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x45.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x46.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x47.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x48.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x49.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x50.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x51.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x52.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x53.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x54.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x55.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x56.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x57.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x58.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x59.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x60.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x61.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x62.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x63.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x64.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x65.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x66.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x67.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x68.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x69.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x70.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x71.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x72.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x73.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x74.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x75.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x76.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x77.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        x78.1:
          dtype: fp32
          shape: [-1, -1, -1, -1]
        '479':
          dtype: fp32
          shape: [32000, 5120]
           
        '483':
          dtype: s64
          shape: [1]
           
        aten::pow_2539_other:
          dtype: fp32
          shape: [1]
           
        '485':
          dtype: fp32
          shape: [1]
           
        '486':
          dtype: fp32
          shape: [5120]
           
        '487':
          dtype: fp32
          shape: [5120]
           
        '488':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3436_bias:
          dtype: fp32
          shape: [5120]
           
        '489':
          dtype: fp32
          shape: [5120]
           
        '490':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3437_bias:
          dtype: fp32
          shape: [5120]
           
        '491':
          dtype: fp32
          shape: [5120]
           
        '492':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3438_bias:
          dtype: fp32
          shape: [5120]
           
        '493':
          dtype: fp32
          shape: [1, 1, 2048, 128]
           
        '494':
          dtype: fp32
          shape: [1, 1, 2048, 128]
           
        '495':
          dtype: s64
          shape: [1]
           
        aten::neg_4508_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4510_mul_val:
          dtype: fp32
          shape: [1]
           
        '496':
          dtype: fp32
          shape: [1]
           
        '497':
          dtype: fp32
          shape: [1]
           
        '498':
          dtype: fp32
          shape: [5120]
           
        '499':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3439_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2550_other:
          dtype: fp32
          shape: [1]
           
        '500':
          dtype: fp32
          shape: [5120]
           
        '501':
          dtype: fp32
          shape: [5120]
           
        '502':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3440_bias:
          dtype: fp32
          shape: [13824]
           
        '503':
          dtype: fp32
          shape: [5120]
           
        '504':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3441_bias:
          dtype: fp32
          shape: [13824]
           
        '505':
          dtype: fp32
          shape: [13824]
           
        '506':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3442_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2551_other:
          dtype: fp32
          shape: [1]
           
        '507':
          dtype: fp32
          shape: [5120]
           
        '508':
          dtype: fp32
          shape: [5120]
           
        '509':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3443_bias:
          dtype: fp32
          shape: [5120]
           
        '510':
          dtype: fp32
          shape: [5120]
           
        '511':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3444_bias:
          dtype: fp32
          shape: [5120]
           
        '512':
          dtype: fp32
          shape: [5120]
           
        '513':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3445_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4540_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4542_mul_val:
          dtype: fp32
          shape: [1]
           
        '514':
          dtype: fp32
          shape: [5120]
           
        '515':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3446_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2563_other:
          dtype: fp32
          shape: [1]
           
        '516':
          dtype: fp32
          shape: [5120]
           
        '517':
          dtype: fp32
          shape: [5120]
           
        '518':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3447_bias:
          dtype: fp32
          shape: [13824]
           
        '519':
          dtype: fp32
          shape: [5120]
           
        '520':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3448_bias:
          dtype: fp32
          shape: [13824]
           
        '521':
          dtype: fp32
          shape: [13824]
           
        '522':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3449_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2564_other:
          dtype: fp32
          shape: [1]
           
        '523':
          dtype: fp32
          shape: [5120]
           
        '524':
          dtype: fp32
          shape: [5120]
           
        '525':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3450_bias:
          dtype: fp32
          shape: [5120]
           
        '526':
          dtype: fp32
          shape: [5120]
           
        '527':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3451_bias:
          dtype: fp32
          shape: [5120]
           
        '528':
          dtype: fp32
          shape: [5120]
           
        '529':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3452_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4572_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4574_mul_val:
          dtype: fp32
          shape: [1]
           
        '530':
          dtype: fp32
          shape: [5120]
           
        '531':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3453_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2576_other:
          dtype: fp32
          shape: [1]
           
        '532':
          dtype: fp32
          shape: [5120]
           
        '533':
          dtype: fp32
          shape: [5120]
           
        '534':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3454_bias:
          dtype: fp32
          shape: [13824]
           
        '535':
          dtype: fp32
          shape: [5120]
           
        '536':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3455_bias:
          dtype: fp32
          shape: [13824]
           
        '537':
          dtype: fp32
          shape: [13824]
           
        '538':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3456_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2577_other:
          dtype: fp32
          shape: [1]
           
        '539':
          dtype: fp32
          shape: [5120]
           
        '540':
          dtype: fp32
          shape: [5120]
           
        '541':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3457_bias:
          dtype: fp32
          shape: [5120]
           
        '542':
          dtype: fp32
          shape: [5120]
           
        '543':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3458_bias:
          dtype: fp32
          shape: [5120]
           
        '544':
          dtype: fp32
          shape: [5120]
           
        '545':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3459_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4604_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4606_mul_val:
          dtype: fp32
          shape: [1]
           
        '546':
          dtype: fp32
          shape: [5120]
           
        '547':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3460_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2589_other:
          dtype: fp32
          shape: [1]
           
        '548':
          dtype: fp32
          shape: [5120]
           
        '549':
          dtype: fp32
          shape: [5120]
           
        '550':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3461_bias:
          dtype: fp32
          shape: [13824]
           
        '551':
          dtype: fp32
          shape: [5120]
           
        '552':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3462_bias:
          dtype: fp32
          shape: [13824]
           
        '553':
          dtype: fp32
          shape: [13824]
           
        '554':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3463_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2590_other:
          dtype: fp32
          shape: [1]
           
        '555':
          dtype: fp32
          shape: [5120]
           
        '556':
          dtype: fp32
          shape: [5120]
           
        '557':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3464_bias:
          dtype: fp32
          shape: [5120]
           
        '558':
          dtype: fp32
          shape: [5120]
           
        '559':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3465_bias:
          dtype: fp32
          shape: [5120]
           
        '560':
          dtype: fp32
          shape: [5120]
           
        '561':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3466_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4636_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4638_mul_val:
          dtype: fp32
          shape: [1]
           
        '562':
          dtype: fp32
          shape: [5120]
           
        '563':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3467_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2602_other:
          dtype: fp32
          shape: [1]
           
        '564':
          dtype: fp32
          shape: [5120]
           
        '565':
          dtype: fp32
          shape: [5120]
           
        '566':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3468_bias:
          dtype: fp32
          shape: [13824]
           
        '567':
          dtype: fp32
          shape: [5120]
           
        '568':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3469_bias:
          dtype: fp32
          shape: [13824]
           
        '569':
          dtype: fp32
          shape: [13824]
           
        '570':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3470_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2603_other:
          dtype: fp32
          shape: [1]
           
        '571':
          dtype: fp32
          shape: [5120]
           
        '572':
          dtype: fp32
          shape: [5120]
           
        '573':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3471_bias:
          dtype: fp32
          shape: [5120]
           
        '574':
          dtype: fp32
          shape: [5120]
           
        '575':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3472_bias:
          dtype: fp32
          shape: [5120]
           
        '576':
          dtype: fp32
          shape: [5120]
           
        '577':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3473_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4668_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4670_mul_val:
          dtype: fp32
          shape: [1]
           
        '578':
          dtype: fp32
          shape: [5120]
           
        '579':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3474_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2615_other:
          dtype: fp32
          shape: [1]
           
        '580':
          dtype: fp32
          shape: [5120]
           
        '581':
          dtype: fp32
          shape: [5120]
           
        '582':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3475_bias:
          dtype: fp32
          shape: [13824]
           
        '583':
          dtype: fp32
          shape: [5120]
           
        '584':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3476_bias:
          dtype: fp32
          shape: [13824]
           
        '585':
          dtype: fp32
          shape: [13824]
           
        '586':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3477_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2616_other:
          dtype: fp32
          shape: [1]
           
        '587':
          dtype: fp32
          shape: [5120]
           
        '588':
          dtype: fp32
          shape: [5120]
           
        '589':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3478_bias:
          dtype: fp32
          shape: [5120]
           
        '590':
          dtype: fp32
          shape: [5120]
           
        '591':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3479_bias:
          dtype: fp32
          shape: [5120]
           
        '592':
          dtype: fp32
          shape: [5120]
           
        '593':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3480_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4700_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4702_mul_val:
          dtype: fp32
          shape: [1]
           
        '594':
          dtype: fp32
          shape: [5120]
           
        '595':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3481_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2628_other:
          dtype: fp32
          shape: [1]
           
        '596':
          dtype: fp32
          shape: [5120]
           
        '597':
          dtype: fp32
          shape: [5120]
           
        '598':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3482_bias:
          dtype: fp32
          shape: [13824]
           
        '599':
          dtype: fp32
          shape: [5120]
           
        '600':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3483_bias:
          dtype: fp32
          shape: [13824]
           
        '601':
          dtype: fp32
          shape: [13824]
           
        '602':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3484_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2629_other:
          dtype: fp32
          shape: [1]
           
        '603':
          dtype: fp32
          shape: [5120]
           
        '604':
          dtype: fp32
          shape: [5120]
           
        '605':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3485_bias:
          dtype: fp32
          shape: [5120]
           
        '606':
          dtype: fp32
          shape: [5120]
           
        '607':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3486_bias:
          dtype: fp32
          shape: [5120]
           
        '608':
          dtype: fp32
          shape: [5120]
           
        '609':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3487_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4732_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4734_mul_val:
          dtype: fp32
          shape: [1]
           
        '610':
          dtype: fp32
          shape: [5120]
           
        '611':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3488_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2641_other:
          dtype: fp32
          shape: [1]
           
        '612':
          dtype: fp32
          shape: [5120]
           
        '613':
          dtype: fp32
          shape: [5120]
           
        '614':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3489_bias:
          dtype: fp32
          shape: [13824]
           
        '615':
          dtype: fp32
          shape: [5120]
           
        '616':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3490_bias:
          dtype: fp32
          shape: [13824]
           
        '617':
          dtype: fp32
          shape: [13824]
           
        '618':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3491_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2642_other:
          dtype: fp32
          shape: [1]
           
        '619':
          dtype: fp32
          shape: [5120]
           
        '620':
          dtype: fp32
          shape: [5120]
           
        '621':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3492_bias:
          dtype: fp32
          shape: [5120]
           
        '622':
          dtype: fp32
          shape: [5120]
           
        '623':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3493_bias:
          dtype: fp32
          shape: [5120]
           
        '624':
          dtype: fp32
          shape: [5120]
           
        '625':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3494_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4764_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4766_mul_val:
          dtype: fp32
          shape: [1]
           
        '626':
          dtype: fp32
          shape: [5120]
           
        '627':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3495_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2654_other:
          dtype: fp32
          shape: [1]
           
        '628':
          dtype: fp32
          shape: [5120]
           
        '629':
          dtype: fp32
          shape: [5120]
           
        '630':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3496_bias:
          dtype: fp32
          shape: [13824]
           
        '631':
          dtype: fp32
          shape: [5120]
           
        '632':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3497_bias:
          dtype: fp32
          shape: [13824]
           
        '633':
          dtype: fp32
          shape: [13824]
           
        '634':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3498_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2655_other:
          dtype: fp32
          shape: [1]
           
        '635':
          dtype: fp32
          shape: [5120]
           
        '636':
          dtype: fp32
          shape: [5120]
           
        '637':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3499_bias:
          dtype: fp32
          shape: [5120]
           
        '638':
          dtype: fp32
          shape: [5120]
           
        '639':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3500_bias:
          dtype: fp32
          shape: [5120]
           
        '640':
          dtype: fp32
          shape: [5120]
           
        '641':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3501_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4796_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4798_mul_val:
          dtype: fp32
          shape: [1]
           
        '642':
          dtype: fp32
          shape: [5120]
           
        '643':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3502_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2667_other:
          dtype: fp32
          shape: [1]
           
        '644':
          dtype: fp32
          shape: [5120]
           
        '645':
          dtype: fp32
          shape: [5120]
           
        '646':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3503_bias:
          dtype: fp32
          shape: [13824]
           
        '647':
          dtype: fp32
          shape: [5120]
           
        '648':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3504_bias:
          dtype: fp32
          shape: [13824]
           
        '649':
          dtype: fp32
          shape: [13824]
           
        '650':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3505_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2668_other:
          dtype: fp32
          shape: [1]
           
        '651':
          dtype: fp32
          shape: [5120]
           
        '652':
          dtype: fp32
          shape: [5120]
           
        '653':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3506_bias:
          dtype: fp32
          shape: [5120]
           
        '654':
          dtype: fp32
          shape: [5120]
           
        '655':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3507_bias:
          dtype: fp32
          shape: [5120]
           
        '656':
          dtype: fp32
          shape: [5120]
           
        '657':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3508_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4828_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4830_mul_val:
          dtype: fp32
          shape: [1]
           
        '658':
          dtype: fp32
          shape: [5120]
           
        '659':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3509_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2680_other:
          dtype: fp32
          shape: [1]
           
        '660':
          dtype: fp32
          shape: [5120]
           
        '661':
          dtype: fp32
          shape: [5120]
           
        '662':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3510_bias:
          dtype: fp32
          shape: [13824]
           
        '663':
          dtype: fp32
          shape: [5120]
           
        '664':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3511_bias:
          dtype: fp32
          shape: [13824]
           
        '665':
          dtype: fp32
          shape: [13824]
           
        '666':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3512_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2681_other:
          dtype: fp32
          shape: [1]
           
        '667':
          dtype: fp32
          shape: [5120]
           
        '668':
          dtype: fp32
          shape: [5120]
           
        '669':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3513_bias:
          dtype: fp32
          shape: [5120]
           
        '670':
          dtype: fp32
          shape: [5120]
           
        '671':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3514_bias:
          dtype: fp32
          shape: [5120]
           
        '672':
          dtype: fp32
          shape: [5120]
           
        '673':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3515_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4860_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4862_mul_val:
          dtype: fp32
          shape: [1]
           
        '674':
          dtype: fp32
          shape: [5120]
           
        '675':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3516_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2693_other:
          dtype: fp32
          shape: [1]
           
        '676':
          dtype: fp32
          shape: [5120]
           
        '677':
          dtype: fp32
          shape: [5120]
           
        '678':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3517_bias:
          dtype: fp32
          shape: [13824]
           
        '679':
          dtype: fp32
          shape: [5120]
           
        '680':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3518_bias:
          dtype: fp32
          shape: [13824]
           
        '681':
          dtype: fp32
          shape: [13824]
           
        '682':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3519_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2694_other:
          dtype: fp32
          shape: [1]
           
        '683':
          dtype: fp32
          shape: [5120]
           
        '684':
          dtype: fp32
          shape: [5120]
           
        '685':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3520_bias:
          dtype: fp32
          shape: [5120]
           
        '686':
          dtype: fp32
          shape: [5120]
           
        '687':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3521_bias:
          dtype: fp32
          shape: [5120]
           
        '688':
          dtype: fp32
          shape: [5120]
           
        '689':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3522_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4892_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4894_mul_val:
          dtype: fp32
          shape: [1]
           
        '690':
          dtype: fp32
          shape: [5120]
           
        '691':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3523_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2706_other:
          dtype: fp32
          shape: [1]
           
        '692':
          dtype: fp32
          shape: [5120]
           
        '693':
          dtype: fp32
          shape: [5120]
           
        '694':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3524_bias:
          dtype: fp32
          shape: [13824]
           
        '695':
          dtype: fp32
          shape: [5120]
           
        '696':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3525_bias:
          dtype: fp32
          shape: [13824]
           
        '697':
          dtype: fp32
          shape: [13824]
           
        '698':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3526_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2707_other:
          dtype: fp32
          shape: [1]
           
        '699':
          dtype: fp32
          shape: [5120]
           
        '700':
          dtype: fp32
          shape: [5120]
           
        '701':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3527_bias:
          dtype: fp32
          shape: [5120]
           
        '702':
          dtype: fp32
          shape: [5120]
           
        '703':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3528_bias:
          dtype: fp32
          shape: [5120]
           
        '704':
          dtype: fp32
          shape: [5120]
           
        '705':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3529_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4924_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4926_mul_val:
          dtype: fp32
          shape: [1]
           
        '706':
          dtype: fp32
          shape: [5120]
           
        '707':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3530_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2719_other:
          dtype: fp32
          shape: [1]
           
        '708':
          dtype: fp32
          shape: [5120]
           
        '709':
          dtype: fp32
          shape: [5120]
           
        '710':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3531_bias:
          dtype: fp32
          shape: [13824]
           
        '711':
          dtype: fp32
          shape: [5120]
           
        '712':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3532_bias:
          dtype: fp32
          shape: [13824]
           
        '713':
          dtype: fp32
          shape: [13824]
           
        '714':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3533_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2720_other:
          dtype: fp32
          shape: [1]
           
        '715':
          dtype: fp32
          shape: [5120]
           
        '716':
          dtype: fp32
          shape: [5120]
           
        '717':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3534_bias:
          dtype: fp32
          shape: [5120]
           
        '718':
          dtype: fp32
          shape: [5120]
           
        '719':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3535_bias:
          dtype: fp32
          shape: [5120]
           
        '720':
          dtype: fp32
          shape: [5120]
           
        '721':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3536_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4956_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4958_mul_val:
          dtype: fp32
          shape: [1]
           
        '722':
          dtype: fp32
          shape: [5120]
           
        '723':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3537_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2732_other:
          dtype: fp32
          shape: [1]
           
        '724':
          dtype: fp32
          shape: [5120]
           
        '725':
          dtype: fp32
          shape: [5120]
           
        '726':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3538_bias:
          dtype: fp32
          shape: [13824]
           
        '727':
          dtype: fp32
          shape: [5120]
           
        '728':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3539_bias:
          dtype: fp32
          shape: [13824]
           
        '729':
          dtype: fp32
          shape: [13824]
           
        '730':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3540_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2733_other:
          dtype: fp32
          shape: [1]
           
        '731':
          dtype: fp32
          shape: [5120]
           
        '732':
          dtype: fp32
          shape: [5120]
           
        '733':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3541_bias:
          dtype: fp32
          shape: [5120]
           
        '734':
          dtype: fp32
          shape: [5120]
           
        '735':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3542_bias:
          dtype: fp32
          shape: [5120]
           
        '736':
          dtype: fp32
          shape: [5120]
           
        '737':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3543_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_4988_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_4990_mul_val:
          dtype: fp32
          shape: [1]
           
        '738':
          dtype: fp32
          shape: [5120]
           
        '739':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3544_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2745_other:
          dtype: fp32
          shape: [1]
           
        '740':
          dtype: fp32
          shape: [5120]
           
        '741':
          dtype: fp32
          shape: [5120]
           
        '742':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3545_bias:
          dtype: fp32
          shape: [13824]
           
        '743':
          dtype: fp32
          shape: [5120]
           
        '744':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3546_bias:
          dtype: fp32
          shape: [13824]
           
        '745':
          dtype: fp32
          shape: [13824]
           
        '746':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3547_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2746_other:
          dtype: fp32
          shape: [1]
           
        '747':
          dtype: fp32
          shape: [5120]
           
        '748':
          dtype: fp32
          shape: [5120]
           
        '749':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3548_bias:
          dtype: fp32
          shape: [5120]
           
        '750':
          dtype: fp32
          shape: [5120]
           
        '751':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3549_bias:
          dtype: fp32
          shape: [5120]
           
        '752':
          dtype: fp32
          shape: [5120]
           
        '753':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3550_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5020_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5022_mul_val:
          dtype: fp32
          shape: [1]
           
        '754':
          dtype: fp32
          shape: [5120]
           
        '755':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3551_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2758_other:
          dtype: fp32
          shape: [1]
           
        '756':
          dtype: fp32
          shape: [5120]
           
        '757':
          dtype: fp32
          shape: [5120]
           
        '758':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3552_bias:
          dtype: fp32
          shape: [13824]
           
        '759':
          dtype: fp32
          shape: [5120]
           
        '760':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3553_bias:
          dtype: fp32
          shape: [13824]
           
        '761':
          dtype: fp32
          shape: [13824]
           
        '762':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3554_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2759_other:
          dtype: fp32
          shape: [1]
           
        '763':
          dtype: fp32
          shape: [5120]
           
        '764':
          dtype: fp32
          shape: [5120]
           
        '765':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3555_bias:
          dtype: fp32
          shape: [5120]
           
        '766':
          dtype: fp32
          shape: [5120]
           
        '767':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3556_bias:
          dtype: fp32
          shape: [5120]
           
        '768':
          dtype: fp32
          shape: [5120]
           
        '769':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3557_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5052_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5054_mul_val:
          dtype: fp32
          shape: [1]
           
        '770':
          dtype: fp32
          shape: [5120]
           
        '771':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3558_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2771_other:
          dtype: fp32
          shape: [1]
           
        '772':
          dtype: fp32
          shape: [5120]
           
        '773':
          dtype: fp32
          shape: [5120]
           
        '774':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3559_bias:
          dtype: fp32
          shape: [13824]
           
        '775':
          dtype: fp32
          shape: [5120]
           
        '776':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3560_bias:
          dtype: fp32
          shape: [13824]
           
        '777':
          dtype: fp32
          shape: [13824]
           
        '778':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3561_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2772_other:
          dtype: fp32
          shape: [1]
           
        '779':
          dtype: fp32
          shape: [5120]
           
        '780':
          dtype: fp32
          shape: [5120]
           
        '781':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3562_bias:
          dtype: fp32
          shape: [5120]
           
        '782':
          dtype: fp32
          shape: [5120]
           
        '783':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3563_bias:
          dtype: fp32
          shape: [5120]
           
        '784':
          dtype: fp32
          shape: [5120]
           
        '785':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3564_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5084_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5086_mul_val:
          dtype: fp32
          shape: [1]
           
        '786':
          dtype: fp32
          shape: [5120]
           
        '787':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3565_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2784_other:
          dtype: fp32
          shape: [1]
           
        '788':
          dtype: fp32
          shape: [5120]
           
        '789':
          dtype: fp32
          shape: [5120]
           
        '790':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3566_bias:
          dtype: fp32
          shape: [13824]
           
        '791':
          dtype: fp32
          shape: [5120]
           
        '792':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3567_bias:
          dtype: fp32
          shape: [13824]
           
        '793':
          dtype: fp32
          shape: [13824]
           
        '794':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3568_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2785_other:
          dtype: fp32
          shape: [1]
           
        '795':
          dtype: fp32
          shape: [5120]
           
        '796':
          dtype: fp32
          shape: [5120]
           
        '797':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3569_bias:
          dtype: fp32
          shape: [5120]
           
        '798':
          dtype: fp32
          shape: [5120]
           
        '799':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3570_bias:
          dtype: fp32
          shape: [5120]
           
        '800':
          dtype: fp32
          shape: [5120]
           
        '801':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3571_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5116_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5118_mul_val:
          dtype: fp32
          shape: [1]
           
        '802':
          dtype: fp32
          shape: [5120]
           
        '803':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3572_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2797_other:
          dtype: fp32
          shape: [1]
           
        '804':
          dtype: fp32
          shape: [5120]
           
        '805':
          dtype: fp32
          shape: [5120]
           
        '806':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3573_bias:
          dtype: fp32
          shape: [13824]
           
        '807':
          dtype: fp32
          shape: [5120]
           
        '808':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3574_bias:
          dtype: fp32
          shape: [13824]
           
        '809':
          dtype: fp32
          shape: [13824]
           
        '810':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3575_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2798_other:
          dtype: fp32
          shape: [1]
           
        '811':
          dtype: fp32
          shape: [5120]
           
        '812':
          dtype: fp32
          shape: [5120]
           
        '813':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3576_bias:
          dtype: fp32
          shape: [5120]
           
        '814':
          dtype: fp32
          shape: [5120]
           
        '815':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3577_bias:
          dtype: fp32
          shape: [5120]
           
        '816':
          dtype: fp32
          shape: [5120]
           
        '817':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3578_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5148_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5150_mul_val:
          dtype: fp32
          shape: [1]
           
        '818':
          dtype: fp32
          shape: [5120]
           
        '819':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3579_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2810_other:
          dtype: fp32
          shape: [1]
           
        '820':
          dtype: fp32
          shape: [5120]
           
        '821':
          dtype: fp32
          shape: [5120]
           
        '822':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3580_bias:
          dtype: fp32
          shape: [13824]
           
        '823':
          dtype: fp32
          shape: [5120]
           
        '824':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3581_bias:
          dtype: fp32
          shape: [13824]
           
        '825':
          dtype: fp32
          shape: [13824]
           
        '826':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3582_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2811_other:
          dtype: fp32
          shape: [1]
           
        '827':
          dtype: fp32
          shape: [5120]
           
        '828':
          dtype: fp32
          shape: [5120]
           
        '829':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3583_bias:
          dtype: fp32
          shape: [5120]
           
        '830':
          dtype: fp32
          shape: [5120]
           
        '831':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3584_bias:
          dtype: fp32
          shape: [5120]
           
        '832':
          dtype: fp32
          shape: [5120]
           
        '833':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3585_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5180_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5182_mul_val:
          dtype: fp32
          shape: [1]
           
        '834':
          dtype: fp32
          shape: [5120]
           
        '835':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3586_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2823_other:
          dtype: fp32
          shape: [1]
           
        '836':
          dtype: fp32
          shape: [5120]
           
        '837':
          dtype: fp32
          shape: [5120]
           
        '838':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3587_bias:
          dtype: fp32
          shape: [13824]
           
        '839':
          dtype: fp32
          shape: [5120]
           
        '840':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3588_bias:
          dtype: fp32
          shape: [13824]
           
        '841':
          dtype: fp32
          shape: [13824]
           
        '842':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3589_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2824_other:
          dtype: fp32
          shape: [1]
           
        '843':
          dtype: fp32
          shape: [5120]
           
        '844':
          dtype: fp32
          shape: [5120]
           
        '845':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3590_bias:
          dtype: fp32
          shape: [5120]
           
        '846':
          dtype: fp32
          shape: [5120]
           
        '847':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3591_bias:
          dtype: fp32
          shape: [5120]
           
        '848':
          dtype: fp32
          shape: [5120]
           
        '849':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3592_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5212_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5214_mul_val:
          dtype: fp32
          shape: [1]
           
        '850':
          dtype: fp32
          shape: [5120]
           
        '851':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3593_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2836_other:
          dtype: fp32
          shape: [1]
           
        '852':
          dtype: fp32
          shape: [5120]
           
        '853':
          dtype: fp32
          shape: [5120]
           
        '854':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3594_bias:
          dtype: fp32
          shape: [13824]
           
        '855':
          dtype: fp32
          shape: [5120]
           
        '856':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3595_bias:
          dtype: fp32
          shape: [13824]
           
        '857':
          dtype: fp32
          shape: [13824]
           
        '858':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3596_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2837_other:
          dtype: fp32
          shape: [1]
           
        '859':
          dtype: fp32
          shape: [5120]
           
        '860':
          dtype: fp32
          shape: [5120]
           
        '861':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3597_bias:
          dtype: fp32
          shape: [5120]
           
        '862':
          dtype: fp32
          shape: [5120]
           
        '863':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3598_bias:
          dtype: fp32
          shape: [5120]
           
        '864':
          dtype: fp32
          shape: [5120]
           
        '865':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3599_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5244_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5246_mul_val:
          dtype: fp32
          shape: [1]
           
        '866':
          dtype: fp32
          shape: [5120]
           
        '867':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3600_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2849_other:
          dtype: fp32
          shape: [1]
           
        '868':
          dtype: fp32
          shape: [5120]
           
        '869':
          dtype: fp32
          shape: [5120]
           
        '870':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3601_bias:
          dtype: fp32
          shape: [13824]
           
        '871':
          dtype: fp32
          shape: [5120]
           
        '872':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3602_bias:
          dtype: fp32
          shape: [13824]
           
        '873':
          dtype: fp32
          shape: [13824]
           
        '874':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3603_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2850_other:
          dtype: fp32
          shape: [1]
           
        '875':
          dtype: fp32
          shape: [5120]
           
        '876':
          dtype: fp32
          shape: [5120]
           
        '877':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3604_bias:
          dtype: fp32
          shape: [5120]
           
        '878':
          dtype: fp32
          shape: [5120]
           
        '879':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3605_bias:
          dtype: fp32
          shape: [5120]
           
        '880':
          dtype: fp32
          shape: [5120]
           
        '881':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3606_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5276_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5278_mul_val:
          dtype: fp32
          shape: [1]
           
        '882':
          dtype: fp32
          shape: [5120]
           
        '883':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3607_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2862_other:
          dtype: fp32
          shape: [1]
           
        '884':
          dtype: fp32
          shape: [5120]
           
        '885':
          dtype: fp32
          shape: [5120]
           
        '886':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3608_bias:
          dtype: fp32
          shape: [13824]
           
        '887':
          dtype: fp32
          shape: [5120]
           
        '888':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3609_bias:
          dtype: fp32
          shape: [13824]
           
        '889':
          dtype: fp32
          shape: [13824]
           
        '890':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3610_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2863_other:
          dtype: fp32
          shape: [1]
           
        '891':
          dtype: fp32
          shape: [5120]
           
        '892':
          dtype: fp32
          shape: [5120]
           
        '893':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3611_bias:
          dtype: fp32
          shape: [5120]
           
        '894':
          dtype: fp32
          shape: [5120]
           
        '895':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3612_bias:
          dtype: fp32
          shape: [5120]
           
        '896':
          dtype: fp32
          shape: [5120]
           
        '897':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3613_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5308_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5310_mul_val:
          dtype: fp32
          shape: [1]
           
        '898':
          dtype: fp32
          shape: [5120]
           
        '899':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3614_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2875_other:
          dtype: fp32
          shape: [1]
           
        '900':
          dtype: fp32
          shape: [5120]
           
        '901':
          dtype: fp32
          shape: [5120]
           
        '902':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3615_bias:
          dtype: fp32
          shape: [13824]
           
        '903':
          dtype: fp32
          shape: [5120]
           
        '904':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3616_bias:
          dtype: fp32
          shape: [13824]
           
        '905':
          dtype: fp32
          shape: [13824]
           
        '906':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3617_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2876_other:
          dtype: fp32
          shape: [1]
           
        '907':
          dtype: fp32
          shape: [5120]
           
        '908':
          dtype: fp32
          shape: [5120]
           
        '909':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3618_bias:
          dtype: fp32
          shape: [5120]
           
        '910':
          dtype: fp32
          shape: [5120]
           
        '911':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3619_bias:
          dtype: fp32
          shape: [5120]
           
        '912':
          dtype: fp32
          shape: [5120]
           
        '913':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3620_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5340_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5342_mul_val:
          dtype: fp32
          shape: [1]
           
        '914':
          dtype: fp32
          shape: [5120]
           
        '915':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3621_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2888_other:
          dtype: fp32
          shape: [1]
           
        '916':
          dtype: fp32
          shape: [5120]
           
        '917':
          dtype: fp32
          shape: [5120]
           
        '918':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3622_bias:
          dtype: fp32
          shape: [13824]
           
        '919':
          dtype: fp32
          shape: [5120]
           
        '920':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3623_bias:
          dtype: fp32
          shape: [13824]
           
        '921':
          dtype: fp32
          shape: [13824]
           
        '922':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3624_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2889_other:
          dtype: fp32
          shape: [1]
           
        '923':
          dtype: fp32
          shape: [5120]
           
        '924':
          dtype: fp32
          shape: [5120]
           
        '925':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3625_bias:
          dtype: fp32
          shape: [5120]
           
        '926':
          dtype: fp32
          shape: [5120]
           
        '927':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3626_bias:
          dtype: fp32
          shape: [5120]
           
        '928':
          dtype: fp32
          shape: [5120]
           
        '929':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3627_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5372_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5374_mul_val:
          dtype: fp32
          shape: [1]
           
        '930':
          dtype: fp32
          shape: [5120]
           
        '931':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3628_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2901_other:
          dtype: fp32
          shape: [1]
           
        '932':
          dtype: fp32
          shape: [5120]
           
        '933':
          dtype: fp32
          shape: [5120]
           
        '934':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3629_bias:
          dtype: fp32
          shape: [13824]
           
        '935':
          dtype: fp32
          shape: [5120]
           
        '936':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3630_bias:
          dtype: fp32
          shape: [13824]
           
        '937':
          dtype: fp32
          shape: [13824]
           
        '938':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3631_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2902_other:
          dtype: fp32
          shape: [1]
           
        '939':
          dtype: fp32
          shape: [5120]
           
        '940':
          dtype: fp32
          shape: [5120]
           
        '941':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3632_bias:
          dtype: fp32
          shape: [5120]
           
        '942':
          dtype: fp32
          shape: [5120]
           
        '943':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3633_bias:
          dtype: fp32
          shape: [5120]
           
        '944':
          dtype: fp32
          shape: [5120]
           
        '945':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3634_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5404_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5406_mul_val:
          dtype: fp32
          shape: [1]
           
        '946':
          dtype: fp32
          shape: [5120]
           
        '947':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3635_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2914_other:
          dtype: fp32
          shape: [1]
           
        '948':
          dtype: fp32
          shape: [5120]
           
        '949':
          dtype: fp32
          shape: [5120]
           
        '950':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3636_bias:
          dtype: fp32
          shape: [13824]
           
        '951':
          dtype: fp32
          shape: [5120]
           
        '952':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3637_bias:
          dtype: fp32
          shape: [13824]
           
        '953':
          dtype: fp32
          shape: [13824]
           
        '954':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3638_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2915_other:
          dtype: fp32
          shape: [1]
           
        '955':
          dtype: fp32
          shape: [5120]
           
        '956':
          dtype: fp32
          shape: [5120]
           
        '957':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3639_bias:
          dtype: fp32
          shape: [5120]
           
        '958':
          dtype: fp32
          shape: [5120]
           
        '959':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3640_bias:
          dtype: fp32
          shape: [5120]
           
        '960':
          dtype: fp32
          shape: [5120]
           
        '961':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3641_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5436_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5438_mul_val:
          dtype: fp32
          shape: [1]
           
        '962':
          dtype: fp32
          shape: [5120]
           
        '963':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3642_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2927_other:
          dtype: fp32
          shape: [1]
           
        '964':
          dtype: fp32
          shape: [5120]
           
        '965':
          dtype: fp32
          shape: [5120]
           
        '966':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3643_bias:
          dtype: fp32
          shape: [13824]
           
        '967':
          dtype: fp32
          shape: [5120]
           
        '968':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3644_bias:
          dtype: fp32
          shape: [13824]
           
        '969':
          dtype: fp32
          shape: [13824]
           
        '970':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3645_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2928_other:
          dtype: fp32
          shape: [1]
           
        '971':
          dtype: fp32
          shape: [5120]
           
        '972':
          dtype: fp32
          shape: [5120]
           
        '973':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3646_bias:
          dtype: fp32
          shape: [5120]
           
        '974':
          dtype: fp32
          shape: [5120]
           
        '975':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3647_bias:
          dtype: fp32
          shape: [5120]
           
        '976':
          dtype: fp32
          shape: [5120]
           
        '977':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3648_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5468_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5470_mul_val:
          dtype: fp32
          shape: [1]
           
        '978':
          dtype: fp32
          shape: [5120]
           
        '979':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3649_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2940_other:
          dtype: fp32
          shape: [1]
           
        '980':
          dtype: fp32
          shape: [5120]
           
        '981':
          dtype: fp32
          shape: [5120]
           
        '982':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3650_bias:
          dtype: fp32
          shape: [13824]
           
        '983':
          dtype: fp32
          shape: [5120]
           
        '984':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3651_bias:
          dtype: fp32
          shape: [13824]
           
        '985':
          dtype: fp32
          shape: [13824]
           
        '986':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3652_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2941_other:
          dtype: fp32
          shape: [1]
           
        '987':
          dtype: fp32
          shape: [5120]
           
        '988':
          dtype: fp32
          shape: [5120]
           
        '989':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3653_bias:
          dtype: fp32
          shape: [5120]
           
        '990':
          dtype: fp32
          shape: [5120]
           
        '991':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3654_bias:
          dtype: fp32
          shape: [5120]
           
        '992':
          dtype: fp32
          shape: [5120]
           
        '993':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3655_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5500_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5502_mul_val:
          dtype: fp32
          shape: [1]
           
        '994':
          dtype: fp32
          shape: [5120]
           
        '995':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3656_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2953_other:
          dtype: fp32
          shape: [1]
           
        '996':
          dtype: fp32
          shape: [5120]
           
        '997':
          dtype: fp32
          shape: [5120]
           
        '998':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3657_bias:
          dtype: fp32
          shape: [13824]
           
        '999':
          dtype: fp32
          shape: [5120]
           
        '1000':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3658_bias:
          dtype: fp32
          shape: [13824]
           
        '1001':
          dtype: fp32
          shape: [13824]
           
        '1002':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3659_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2954_other:
          dtype: fp32
          shape: [1]
           
        '1003':
          dtype: fp32
          shape: [5120]
           
        '1004':
          dtype: fp32
          shape: [5120]
           
        '1005':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3660_bias:
          dtype: fp32
          shape: [5120]
           
        '1006':
          dtype: fp32
          shape: [5120]
           
        '1007':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3661_bias:
          dtype: fp32
          shape: [5120]
           
        '1008':
          dtype: fp32
          shape: [5120]
           
        '1009':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3662_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5532_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5534_mul_val:
          dtype: fp32
          shape: [1]
           
        '1010':
          dtype: fp32
          shape: [5120]
           
        '1011':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3663_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2966_other:
          dtype: fp32
          shape: [1]
           
        '1012':
          dtype: fp32
          shape: [5120]
           
        '1013':
          dtype: fp32
          shape: [5120]
           
        '1014':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3664_bias:
          dtype: fp32
          shape: [13824]
           
        '1015':
          dtype: fp32
          shape: [5120]
           
        '1016':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3665_bias:
          dtype: fp32
          shape: [13824]
           
        '1017':
          dtype: fp32
          shape: [13824]
           
        '1018':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3666_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2967_other:
          dtype: fp32
          shape: [1]
           
        '1019':
          dtype: fp32
          shape: [5120]
           
        '1020':
          dtype: fp32
          shape: [5120]
           
        '1021':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3667_bias:
          dtype: fp32
          shape: [5120]
           
        '1022':
          dtype: fp32
          shape: [5120]
           
        '1023':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3668_bias:
          dtype: fp32
          shape: [5120]
           
        '1024':
          dtype: fp32
          shape: [5120]
           
        '1025':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3669_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5564_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5566_mul_val:
          dtype: fp32
          shape: [1]
           
        '1026':
          dtype: fp32
          shape: [5120]
           
        '1027':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3670_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2979_other:
          dtype: fp32
          shape: [1]
           
        '1028':
          dtype: fp32
          shape: [5120]
           
        '1029':
          dtype: fp32
          shape: [5120]
           
        '1030':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3671_bias:
          dtype: fp32
          shape: [13824]
           
        '1031':
          dtype: fp32
          shape: [5120]
           
        '1032':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3672_bias:
          dtype: fp32
          shape: [13824]
           
        '1033':
          dtype: fp32
          shape: [13824]
           
        '1034':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3673_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2980_other:
          dtype: fp32
          shape: [1]
           
        '1035':
          dtype: fp32
          shape: [5120]
           
        '1036':
          dtype: fp32
          shape: [5120]
           
        '1037':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3674_bias:
          dtype: fp32
          shape: [5120]
           
        '1038':
          dtype: fp32
          shape: [5120]
           
        '1039':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3675_bias:
          dtype: fp32
          shape: [5120]
           
        '1040':
          dtype: fp32
          shape: [5120]
           
        '1041':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3676_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5596_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5598_mul_val:
          dtype: fp32
          shape: [1]
           
        '1042':
          dtype: fp32
          shape: [5120]
           
        '1043':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3677_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2992_other:
          dtype: fp32
          shape: [1]
           
        '1044':
          dtype: fp32
          shape: [5120]
           
        '1045':
          dtype: fp32
          shape: [5120]
           
        '1046':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3678_bias:
          dtype: fp32
          shape: [13824]
           
        '1047':
          dtype: fp32
          shape: [5120]
           
        '1048':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3679_bias:
          dtype: fp32
          shape: [13824]
           
        '1049':
          dtype: fp32
          shape: [13824]
           
        '1050':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3680_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_2993_other:
          dtype: fp32
          shape: [1]
           
        '1051':
          dtype: fp32
          shape: [5120]
           
        '1052':
          dtype: fp32
          shape: [5120]
           
        '1053':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3681_bias:
          dtype: fp32
          shape: [5120]
           
        '1054':
          dtype: fp32
          shape: [5120]
           
        '1055':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3682_bias:
          dtype: fp32
          shape: [5120]
           
        '1056':
          dtype: fp32
          shape: [5120]
           
        '1057':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3683_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5628_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5630_mul_val:
          dtype: fp32
          shape: [1]
           
        '1058':
          dtype: fp32
          shape: [5120]
           
        '1059':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3684_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3005_other:
          dtype: fp32
          shape: [1]
           
        '1060':
          dtype: fp32
          shape: [5120]
           
        '1061':
          dtype: fp32
          shape: [5120]
           
        '1062':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3685_bias:
          dtype: fp32
          shape: [13824]
           
        '1063':
          dtype: fp32
          shape: [5120]
           
        '1064':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3686_bias:
          dtype: fp32
          shape: [13824]
           
        '1065':
          dtype: fp32
          shape: [13824]
           
        '1066':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3687_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3006_other:
          dtype: fp32
          shape: [1]
           
        '1067':
          dtype: fp32
          shape: [5120]
           
        '1068':
          dtype: fp32
          shape: [5120]
           
        '1069':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3688_bias:
          dtype: fp32
          shape: [5120]
           
        '1070':
          dtype: fp32
          shape: [5120]
           
        '1071':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3689_bias:
          dtype: fp32
          shape: [5120]
           
        '1072':
          dtype: fp32
          shape: [5120]
           
        '1073':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3690_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5660_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5662_mul_val:
          dtype: fp32
          shape: [1]
           
        '1074':
          dtype: fp32
          shape: [5120]
           
        '1075':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3691_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3018_other:
          dtype: fp32
          shape: [1]
           
        '1076':
          dtype: fp32
          shape: [5120]
           
        '1077':
          dtype: fp32
          shape: [5120]
           
        '1078':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3692_bias:
          dtype: fp32
          shape: [13824]
           
        '1079':
          dtype: fp32
          shape: [5120]
           
        '1080':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3693_bias:
          dtype: fp32
          shape: [13824]
           
        '1081':
          dtype: fp32
          shape: [13824]
           
        '1082':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3694_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3019_other:
          dtype: fp32
          shape: [1]
           
        '1083':
          dtype: fp32
          shape: [5120]
           
        '1084':
          dtype: fp32
          shape: [5120]
           
        '1085':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3695_bias:
          dtype: fp32
          shape: [5120]
           
        '1086':
          dtype: fp32
          shape: [5120]
           
        '1087':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3696_bias:
          dtype: fp32
          shape: [5120]
           
        '1088':
          dtype: fp32
          shape: [5120]
           
        '1089':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3697_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5692_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5694_mul_val:
          dtype: fp32
          shape: [1]
           
        '1090':
          dtype: fp32
          shape: [5120]
           
        '1091':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3698_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3031_other:
          dtype: fp32
          shape: [1]
           
        '1092':
          dtype: fp32
          shape: [5120]
           
        '1093':
          dtype: fp32
          shape: [5120]
           
        '1094':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3699_bias:
          dtype: fp32
          shape: [13824]
           
        '1095':
          dtype: fp32
          shape: [5120]
           
        '1096':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3700_bias:
          dtype: fp32
          shape: [13824]
           
        '1097':
          dtype: fp32
          shape: [13824]
           
        '1098':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3701_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3032_other:
          dtype: fp32
          shape: [1]
           
        '1099':
          dtype: fp32
          shape: [5120]
           
        '1100':
          dtype: fp32
          shape: [5120]
           
        '1101':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3702_bias:
          dtype: fp32
          shape: [5120]
           
        '1102':
          dtype: fp32
          shape: [5120]
           
        '1103':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3703_bias:
          dtype: fp32
          shape: [5120]
           
        '1104':
          dtype: fp32
          shape: [5120]
           
        '1105':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3704_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5724_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5726_mul_val:
          dtype: fp32
          shape: [1]
           
        '1106':
          dtype: fp32
          shape: [5120]
           
        '1107':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3705_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3044_other:
          dtype: fp32
          shape: [1]
           
        '1108':
          dtype: fp32
          shape: [5120]
           
        '1109':
          dtype: fp32
          shape: [5120]
           
        '1110':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3706_bias:
          dtype: fp32
          shape: [13824]
           
        '1111':
          dtype: fp32
          shape: [5120]
           
        '1112':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3707_bias:
          dtype: fp32
          shape: [13824]
           
        '1113':
          dtype: fp32
          shape: [13824]
           
        '1114':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3708_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3045_other:
          dtype: fp32
          shape: [1]
           
        '1115':
          dtype: fp32
          shape: [5120]
           
        '1116':
          dtype: fp32
          shape: [5120]
           
        '1117':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3709_bias:
          dtype: fp32
          shape: [5120]
           
        '1118':
          dtype: fp32
          shape: [5120]
           
        '1119':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3710_bias:
          dtype: fp32
          shape: [5120]
           
        '1120':
          dtype: fp32
          shape: [5120]
           
        '1121':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3711_bias:
          dtype: fp32
          shape: [5120]
           
        aten::neg_5756_mul_val:
          dtype: fp32
          shape: [1]
           
        aten::neg_5758_mul_val:
          dtype: fp32
          shape: [1]
           
        '1122':
          dtype: fp32
          shape: [5120]
           
        '1123':
          dtype: s8
          shape: [5120, 5120]
           
        aten::linear_3712_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3057_other:
          dtype: fp32
          shape: [1]
           
        '1124':
          dtype: fp32
          shape: [5120]
           
        '1125':
          dtype: fp32
          shape: [5120]
           
        '1126':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3713_bias:
          dtype: fp32
          shape: [13824]
           
        '1127':
          dtype: fp32
          shape: [5120]
           
        '1128':
          dtype: s8
          shape: [13824, 5120]
           
        aten::linear_3714_bias:
          dtype: fp32
          shape: [13824]
           
        '1129':
          dtype: fp32
          shape: [13824]
           
        '1130':
          dtype: s8
          shape: [5120, 13824]
           
        aten::linear_3715_bias:
          dtype: fp32
          shape: [5120]
           
        aten::pow_3058_other:
          dtype: fp32
          shape: [1]
           
        '1131':
          dtype: fp32
          shape: [5120]
           
        '1132':
          dtype: fp32
          shape: [5120]
           
        '1133':
          dtype: s8
          shape: [32000, 5120]
           
        aten::linear_3716_bias:
          dtype: fp32
          shape: [32000]
           
    aten::size_0:
      type: Shape
      input:
        input_ids.1: {}
      output:
        '1336': {}
      attr:
        start: 0
        end: 0
    aten::size_1:
      type: Shape
      input:
        input_ids.1: {}
      output:
        '1338': {}
      attr:
        start: 1
        end: 1
    aten::size_2537:
      type: Shape
      input:
        x.1: {}
      output:
        '1344': {}
      attr:
        start: 2
        end: 2
    aten::embedding_2:
      type: Gather
      input:
        '479': {}
        input_ids.1: {}
      output:
        ret2.1: {}
    aten::full_2534:
      type: Full
      input:
        '1338': {}
      output:
        mask.1: {}
    aten::size_2450:
      type: Shape
      input:
        mask.1: {}
      output:
        '1352': {}
      attr:
        start: -1
        end: -1
    aten::arange_3432:
      type: Arange
      input:
        '1352': {}
      output:
        mask_cond.1: {}
    aten::add_7:
      type: Add
      input:
        mask_cond.1: {}
        '483': {}
      output:
        '1354': {}
    aten::size_2451:
      type: Shape
      input:
        mask.1: {}
      output:
        '1355': {}
      attr:
        start: -1
        end: -1
    aten::view_4494:
      type: View
      input:
        '1354': {}
        '1355': {}
      output:
        '1357': {}
      attr:
        shape: -1,1
    aten::lt_4493:
      type: Less
      input:
        mask_cond.1: {}
        '1357': {}
      output:
        '1358': {}
      attr:
        algorithm: lt
    aten::masked_fill_3429:
      type: ConstantOfShape
      input:
        mask.1: {}
        '1358': {}
      output:
        mask0.1: {}
    aten::zeros_2408:
      type: Zeros
      input:
        '1338': {}
        '1344': {}
      output:
        '1362': {}
    aten::cat_2452:
      type: Concat
      input:
        '1362': {}
        mask0.1: {}
      output:
        mask2.1: {}
      attr:
        axis: -1
    aten::unsqueeze_3388:
      type: Unsqueeze
      input:
        mask2.1: {}
      output:
        '1365': {}
      attr:
        axes: 0
    aten::unsqueeze_3061:
      type: Unsqueeze
      input:
        '1365': {}
      output:
        '1366': {}
      attr:
        axes: 1
    aten::slice_2406:
      type: Slice
      input:
        '1366': {}
      output:
        '1367': {}
      attr:
        axes: 2
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2043:
      type: Slice
      input:
        '1367': {}
      output:
        '1368': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::add_3062:
      type: Add
      input:
        '1338': {}
        '1344': {}
      output:
        '1371': {}
    aten::expand_3433:
      type: Expand
      input:
        '1368': {}
        '1336': {}
        '1338': {}
        '1371': {}
      output:
        '1374': {}
    aten::size_3:
      type: Shape
      input:
        attention_mask.1: {}
      output:
        '1376': {}
      attr:
        start: 0
        end: 0
    aten::size_4:
      type: Shape
      input:
        attention_mask.1: {}
      output:
        '1378': {}
      attr:
        start: 1
        end: 1
    aten::slice_5:
      type: Slice
      input:
        attention_mask.1: {}
      output:
        '1384': {}
      attr:
        axes: 0
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::unsqueeze_3064:
      type: Unsqueeze
      input:
        '1384': {}
      output:
        '1385': {}
      attr:
        axes: 1
    aten::unsqueeze_2538:
      type: Unsqueeze
      input:
        '1385': {}
      output:
        '1386': {}
      attr:
        axes: 2
    aten::slice_2044:
      type: Slice
      input:
        '1386': {}
      output:
        ret5.1: {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::expand_3434:
      type: Expand
      input:
        ret5.1: {}
        '1376': {}
        '1338': {}
        '1378': {}
      output:
        ret6.1: {}
    aten::rsub_2040:
      type: Rsub
      input:
        ret6.1: {}
      output:
        ret7.1: {}
      attr:
        other: 1.0
        alpha: 1
    aten::masked_fill_2535:
      type: ConstantOfShape
      input:
        ret7.1: {}
      output:
        ret9.1: {}
    aten::add_3066:
      type: Add
      input:
        ret9.1: {}
        '1374': {}
      output:
        attention_mask0.1: {}
    aten::pow_2539:
      type: Pow
      input:
        ret2.1: {}
        aten::pow_2539_other: {}
      output:
        ret10.1: {}
    aten::mean_973:
      type: ReduceMean
      input:
        ret10.1: {}
      output:
        ret11.1: {}
    aten::add_8:
      type: Add
      input:
        ret11.1: {}
        '485': {}
      output:
        ret12.1: {}
    aten::rsqrt_4496:
      type: Rsqrt
      input:
        ret12.1: {}
      output:
        ret13.1: {}
    aten::mul_4492:
      type: Mul
      input:
        ret2.1: {}
        ret13.1: {}
      output:
        ret14.1: {}
      attr:
        algorithm: mul
    aten::mul_89:
      type: Mul
      input:
        '486': {}
        ret14.1: {}
      output:
        ret15.1: {}
      attr:
        algorithm: mul
    aten::size_3389:
      type: Shape
      input:
        ret15.1: {}
      output:
        '1416': {}
      attr:
        start: 0
        end: 0
    aten::size_3067:
      type: Shape
      input:
        ret15.1: {}
      output:
        '1418': {}
      attr:
        start: 1
        end: 1
    aten::mul_90:
      type: Mul
      input:
        ret15.1: {}
        '487': {}
      output:
        ret18.1: {}
      attr:
        algorithm: mul
    aten::linear_3436:
      type: InnerProduct
      input:
        ret18.1: {}
        '488': {}
        aten::linear_3436_bias: {}
      output:
        ret21.1: {}
    aten::view_4498:
      type: View
      input:
        ret21.1: {}
        '1416': {}
        '1418': {}
      output:
        ret22.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2540:
      type: Reorder
      input:
        ret22.1: {}
      output:
        ret23.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_92:
      type: Mul
      input:
        ret15.1: {}
        '489': {}
      output:
        ret24.1: {}
      attr:
        algorithm: mul
    aten::linear_3437:
      type: InnerProduct
      input:
        ret24.1: {}
        '490': {}
        aten::linear_3437_bias: {}
      output:
        ret27.1: {}
    aten::view_4499:
      type: View
      input:
        ret27.1: {}
        '1416': {}
        '1418': {}
      output:
        ret28.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2541:
      type: Reorder
      input:
        ret28.1: {}
      output:
        ret29.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_94:
      type: Mul
      input:
        ret15.1: {}
        '491': {}
      output:
        ret30.1: {}
      attr:
        algorithm: mul
    aten::linear_3438:
      type: InnerProduct
      input:
        ret30.1: {}
        '492': {}
        aten::linear_3438_bias: {}
      output:
        ret33.1: {}
    aten::view_4500:
      type: View
      input:
        ret33.1: {}
        '1416': {}
        '1418': {}
      output:
        ret34.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2542:
      type: Reorder
      input:
        ret34.1: {}
      output:
        ret35.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2543:
      type: Shape
      input:
        ret29.1: {}
      output:
        '1464': {}
      attr:
        start: 2
        end: 2
    aten::add_3347:
      type: Add
      input:
        '1464': {}
        '1344': {}
      output:
        seq_len.1: {}
    aten::slice_96:
      type: Slice
      input:
        '493': {}
        seq_len.1: {}
      output:
        '1470': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_136:
      type: Slice
      input:
        '494': {}
        seq_len.1: {}
      output:
        '1472': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2544:
      type: Shape
      input:
        ret23.1: {}
      output:
        '1474': {}
      attr:
        start: 2
        end: 2
    aten::add_3068:
      type: Add
      input:
        '1474': {}
        '1344': {}
      output:
        '1478': {}
    aten::slice_2545:
      type: Slice
      input:
        '1470': {}
        '1344': {}
        '1478': {}
      output:
        '1480': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2045:
      type: Slice
      input:
        '1480': {}
      output:
        '1481': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2546:
      type: Slice
      input:
        '1472': {}
        '1344': {}
        '1478': {}
      output:
        '1482': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2046:
      type: Slice
      input:
        '1482': {}
      output:
        '1483': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4501:
      type: Mul
      input:
        ret23.1: {}
        '1481': {}
      output:
        ret38.1: {}
      attr:
        algorithm: mul
    aten::size_2047:
      type: Shape
      input:
        ret23.1: {}
      output:
        '1487': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_176:
      type: Div
      input:
        '1487': {}
        '495': {}
      output:
        ret40.1: {}
      attr:
        algorithm: div
    aten::slice_2048:
      type: Slice
      input:
        ret23.1: {}
        ret40.1: {}
      output:
        ret41.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2049:
      type: Slice
      input:
        ret23.1: {}
        ret40.1: {}
      output:
        ret42.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4508:
      type: Neg
      input:
        ret42.1: {}
        aten::neg_4508_mul_val: {}
      output:
        ret43.1: {}
      attr:
        algorithm: mul
    aten::cat_2453:
      type: Concat
      input:
        ret43.1: {}
        ret41.1: {}
      output:
        ret44.1: {}
      attr:
        axis: -1
    aten::mul_4505:
      type: Mul
      input:
        ret44.1: {}
        '1483': {}
      output:
        ret45.1: {}
      attr:
        algorithm: mul
    aten::add_3069:
      type: Add
      input:
        ret38.1: {}
        ret45.1: {}
      output:
        args.1: {}
    aten::mul_4503:
      type: Mul
      input:
        ret29.1: {}
        '1481': {}
      output:
        ret46.1: {}
      attr:
        algorithm: mul
    aten::size_2050:
      type: Shape
      input:
        ret29.1: {}
      output:
        '1509': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_177:
      type: Div
      input:
        '1509': {}
        '495': {}
      output:
        ret48.1: {}
      attr:
        algorithm: div
    aten::slice_2051:
      type: Slice
      input:
        ret29.1: {}
        ret48.1: {}
      output:
        ret49.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2052:
      type: Slice
      input:
        ret29.1: {}
        ret48.1: {}
      output:
        ret50.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4510:
      type: Neg
      input:
        ret50.1: {}
        aten::neg_4510_mul_val: {}
      output:
        ret51.1: {}
      attr:
        algorithm: mul
    aten::cat_2454:
      type: Concat
      input:
        ret51.1: {}
        ret49.1: {}
      output:
        ret52.1: {}
      attr:
        axis: -1
    aten::mul_4506:
      type: Mul
      input:
        ret52.1: {}
        '1483': {}
      output:
        ret53.1: {}
      attr:
        algorithm: mul
    aten::add_3070:
      type: Add
      input:
        ret46.1: {}
        ret53.1: {}
      output:
        '1527': {}
    aten::cat_2547:
      type: Concat
      input:
        x.1: {}
        '1527': {}
      output:
        ret54.1: {}
      attr:
        axis: 2
    aten::cat_2548:
      type: Concat
      input:
        x0.1: {}
        ret35.1: {}
      output:
        ret55.1: {}
      attr:
        axis: 2
    aten::transpose_2053:
      type: Reorder
      input:
        ret54.1: {}
      output:
        ret56.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4513:
      type: Matmul
      input:
        args.1: {}
        ret56.1: {}
      output:
        ret59.1: {}
    aten::div_256:
      type: Div
      input:
        ret59.1: {}
        '496': {}
      output:
        ret60.1: {}
      attr:
        algorithm: div
    aten::add_3071:
      type: Add
      input:
        ret60.1: {}
        attention_mask0.1: {}
      output:
        attn_weights0.1: {}
    aten::max_296:
      type: Max
      input:
        attn_weights0.1: {}
        '497': {}
      output:
        input0.1: {}
    aten::softmax_2409:
      type: Softmax
      input:
        input0.1: {}
      output:
        '1550': {}
      attr:
        axis: -1
    aten::matmul_4516:
      type: Matmul
      input:
        '1550': {}
        ret55.1: {}
      output:
        ret63.1: {}
    aten::transpose_2549:
      type: Reorder
      input:
        ret63.1: {}
      output:
        ret64.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4518:
      type: Reshape
      input:
        ret64.1: {}
        '1416': {}
        '1418': {}
      output:
        ret65.1: {}
    aten::mul_336:
      type: Mul
      input:
        ret65.1: {}
        '498': {}
      output:
        ret66.1: {}
      attr:
        algorithm: mul
    aten::linear_3439:
      type: InnerProduct
      input:
        ret66.1: {}
        '499': {}
        aten::linear_3439_bias: {}
      output:
        ret69.1: {}
    aten::add_3072:
      type: Add
      input:
        ret2.1: {}
        ret69.1: {}
      output:
        ret70.1: {}
    aten::pow_2550:
      type: Pow
      input:
        ret70.1: {}
        aten::pow_2550_other: {}
      output:
        ret71.1: {}
    aten::mean_974:
      type: ReduceMean
      input:
        ret71.1: {}
      output:
        ret72.1: {}
    aten::add_9:
      type: Add
      input:
        ret72.1: {}
        '485': {}
      output:
        ret73.1: {}
    aten::rsqrt_4521:
      type: Rsqrt
      input:
        ret73.1: {}
      output:
        ret74.1: {}
    aten::mul_4520:
      type: Mul
      input:
        ret70.1: {}
        ret74.1: {}
      output:
        ret75.1: {}
      attr:
        algorithm: mul
    aten::mul_338:
      type: Mul
      input:
        '500': {}
        ret75.1: {}
      output:
        ret76.1: {}
      attr:
        algorithm: mul
    aten::mul_339:
      type: Mul
      input:
        ret76.1: {}
        '501': {}
      output:
        ret77.1: {}
      attr:
        algorithm: mul
    aten::linear_3440:
      type: InnerProduct
      input:
        ret77.1: {}
        '502': {}
        aten::linear_3440_bias: {}
      output:
        ret80.1: {}
    aten::silu_4523:
      type: Swish
      input:
        ret80.1: {}
      output:
        ret81.1: {}
    aten::mul_341:
      type: Mul
      input:
        ret76.1: {}
        '503': {}
      output:
        ret82.1: {}
      attr:
        algorithm: mul
    aten::linear_3441:
      type: InnerProduct
      input:
        ret82.1: {}
        '504': {}
        aten::linear_3441_bias: {}
      output:
        ret85.1: {}
    aten::mul_4524:
      type: Mul
      input:
        ret81.1: {}
        ret85.1: {}
      output:
        ret86.1: {}
      attr:
        algorithm: mul
    aten::mul_343:
      type: Mul
      input:
        ret86.1: {}
        '505': {}
      output:
        ret87.1: {}
      attr:
        algorithm: mul
    aten::linear_3442:
      type: InnerProduct
      input:
        ret87.1: {}
        '506': {}
        aten::linear_3442_bias: {}
      output:
        ret90.1: {}
    aten::add_3073:
      type: Add
      input:
        ret70.1: {}
        ret90.1: {}
      output:
        ret91.1: {}
    aten::pow_2551:
      type: Pow
      input:
        ret91.1: {}
        aten::pow_2551_other: {}
      output:
        ret92.1: {}
    aten::mean_975:
      type: ReduceMean
      input:
        ret92.1: {}
      output:
        ret93.1: {}
    aten::add_10:
      type: Add
      input:
        ret93.1: {}
        '485': {}
      output:
        ret94.1: {}
    aten::rsqrt_4528:
      type: Rsqrt
      input:
        ret94.1: {}
      output:
        ret95.1: {}
    aten::mul_4527:
      type: Mul
      input:
        ret91.1: {}
        ret95.1: {}
      output:
        ret96.1: {}
      attr:
        algorithm: mul
    aten::mul_345:
      type: Mul
      input:
        '507': {}
        ret96.1: {}
      output:
        ret97.1: {}
      attr:
        algorithm: mul
    aten::size_3390:
      type: Shape
      input:
        ret97.1: {}
      output:
        '1635': {}
      attr:
        start: 0
        end: 0
    aten::size_3074:
      type: Shape
      input:
        ret97.1: {}
      output:
        '1637': {}
      attr:
        start: 1
        end: 1
    aten::mul_346:
      type: Mul
      input:
        ret97.1: {}
        '508': {}
      output:
        ret100.1: {}
      attr:
        algorithm: mul
    aten::linear_3443:
      type: InnerProduct
      input:
        ret100.1: {}
        '509': {}
        aten::linear_3443_bias: {}
      output:
        ret103.1: {}
    aten::view_4530:
      type: View
      input:
        ret103.1: {}
        '1635': {}
        '1637': {}
      output:
        ret104.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2552:
      type: Reorder
      input:
        ret104.1: {}
      output:
        ret105.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_348:
      type: Mul
      input:
        ret97.1: {}
        '510': {}
      output:
        ret106.1: {}
      attr:
        algorithm: mul
    aten::linear_3444:
      type: InnerProduct
      input:
        ret106.1: {}
        '511': {}
        aten::linear_3444_bias: {}
      output:
        ret109.1: {}
    aten::view_4531:
      type: View
      input:
        ret109.1: {}
        '1635': {}
        '1637': {}
      output:
        ret110.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2553:
      type: Reorder
      input:
        ret110.1: {}
      output:
        ret111.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_350:
      type: Mul
      input:
        ret97.1: {}
        '512': {}
      output:
        ret112.1: {}
      attr:
        algorithm: mul
    aten::linear_3445:
      type: InnerProduct
      input:
        ret112.1: {}
        '513': {}
        aten::linear_3445_bias: {}
      output:
        ret115.1: {}
    aten::view_4532:
      type: View
      input:
        ret115.1: {}
        '1635': {}
        '1637': {}
      output:
        ret116.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2554:
      type: Reorder
      input:
        ret116.1: {}
      output:
        ret117.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2555:
      type: Shape
      input:
        ret111.1: {}
      output:
        '1683': {}
      attr:
        start: 2
        end: 2
    aten::size_2556:
      type: Shape
      input:
        x1.1: {}
      output:
        '1686': {}
      attr:
        start: 2
        end: 2
    aten::add_3348:
      type: Add
      input:
        '1683': {}
        '1686': {}
      output:
        seq_len0.1: {}
    aten::slice_97:
      type: Slice
      input:
        '493': {}
        seq_len0.1: {}
      output:
        '1693': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_137:
      type: Slice
      input:
        '494': {}
        seq_len0.1: {}
      output:
        '1695': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2557:
      type: Shape
      input:
        ret105.1: {}
      output:
        '1697': {}
      attr:
        start: 2
        end: 2
    aten::add_3075:
      type: Add
      input:
        '1697': {}
        '1686': {}
      output:
        '1702': {}
    aten::slice_2558:
      type: Slice
      input:
        '1693': {}
        '1686': {}
        '1702': {}
      output:
        '1704': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2054:
      type: Slice
      input:
        '1704': {}
      output:
        '1705': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2559:
      type: Slice
      input:
        '1695': {}
        '1686': {}
        '1702': {}
      output:
        '1706': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2055:
      type: Slice
      input:
        '1706': {}
      output:
        '1707': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4533:
      type: Mul
      input:
        ret105.1: {}
        '1705': {}
      output:
        ret121.1: {}
      attr:
        algorithm: mul
    aten::size_2056:
      type: Shape
      input:
        ret105.1: {}
      output:
        '1711': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_178:
      type: Div
      input:
        '1711': {}
        '495': {}
      output:
        ret123.1: {}
      attr:
        algorithm: div
    aten::slice_2057:
      type: Slice
      input:
        ret105.1: {}
        ret123.1: {}
      output:
        ret124.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2058:
      type: Slice
      input:
        ret105.1: {}
        ret123.1: {}
      output:
        ret125.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4540:
      type: Neg
      input:
        ret125.1: {}
        aten::neg_4540_mul_val: {}
      output:
        ret126.1: {}
      attr:
        algorithm: mul
    aten::cat_2455:
      type: Concat
      input:
        ret126.1: {}
        ret124.1: {}
      output:
        ret127.1: {}
      attr:
        axis: -1
    aten::mul_4537:
      type: Mul
      input:
        ret127.1: {}
        '1707': {}
      output:
        ret128.1: {}
      attr:
        algorithm: mul
    aten::add_3076:
      type: Add
      input:
        ret121.1: {}
        ret128.1: {}
      output:
        args3.1: {}
    aten::mul_4535:
      type: Mul
      input:
        ret111.1: {}
        '1705': {}
      output:
        ret129.1: {}
      attr:
        algorithm: mul
    aten::size_2059:
      type: Shape
      input:
        ret111.1: {}
      output:
        '1733': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_179:
      type: Div
      input:
        '1733': {}
        '495': {}
      output:
        ret131.1: {}
      attr:
        algorithm: div
    aten::slice_2060:
      type: Slice
      input:
        ret111.1: {}
        ret131.1: {}
      output:
        ret132.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2061:
      type: Slice
      input:
        ret111.1: {}
        ret131.1: {}
      output:
        ret133.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4542:
      type: Neg
      input:
        ret133.1: {}
        aten::neg_4542_mul_val: {}
      output:
        ret134.1: {}
      attr:
        algorithm: mul
    aten::cat_2456:
      type: Concat
      input:
        ret134.1: {}
        ret132.1: {}
      output:
        ret135.1: {}
      attr:
        axis: -1
    aten::mul_4538:
      type: Mul
      input:
        ret135.1: {}
        '1707': {}
      output:
        ret136.1: {}
      attr:
        algorithm: mul
    aten::add_3077:
      type: Add
      input:
        ret129.1: {}
        ret136.1: {}
      output:
        '1751': {}
    aten::cat_2560:
      type: Concat
      input:
        x1.1: {}
        '1751': {}
      output:
        ret137.1: {}
      attr:
        axis: 2
    aten::cat_2561:
      type: Concat
      input:
        x2.1: {}
        ret117.1: {}
      output:
        ret138.1: {}
      attr:
        axis: 2
    aten::transpose_2062:
      type: Reorder
      input:
        ret137.1: {}
      output:
        ret139.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4545:
      type: Matmul
      input:
        args3.1: {}
        ret139.1: {}
      output:
        ret142.1: {}
    aten::div_257:
      type: Div
      input:
        ret142.1: {}
        '496': {}
      output:
        ret143.1: {}
      attr:
        algorithm: div
    aten::add_3078:
      type: Add
      input:
        ret143.1: {}
        attention_mask0.1: {}
      output:
        attn_weights2.1: {}
    aten::max_297:
      type: Max
      input:
        attn_weights2.1: {}
        '497': {}
      output:
        input2.1: {}
    aten::softmax_2410:
      type: Softmax
      input:
        input2.1: {}
      output:
        '1773': {}
      attr:
        axis: -1
    aten::matmul_4548:
      type: Matmul
      input:
        '1773': {}
        ret138.1: {}
      output:
        ret146.1: {}
    aten::transpose_2562:
      type: Reorder
      input:
        ret146.1: {}
      output:
        ret147.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4550:
      type: Reshape
      input:
        ret147.1: {}
        '1635': {}
        '1637': {}
      output:
        ret148.1: {}
    aten::mul_352:
      type: Mul
      input:
        ret148.1: {}
        '514': {}
      output:
        ret149.1: {}
      attr:
        algorithm: mul
    aten::linear_3446:
      type: InnerProduct
      input:
        ret149.1: {}
        '515': {}
        aten::linear_3446_bias: {}
      output:
        ret152.1: {}
    aten::add_3079:
      type: Add
      input:
        ret91.1: {}
        ret152.1: {}
      output:
        ret153.1: {}
    aten::pow_2563:
      type: Pow
      input:
        ret153.1: {}
        aten::pow_2563_other: {}
      output:
        ret154.1: {}
    aten::mean_976:
      type: ReduceMean
      input:
        ret154.1: {}
      output:
        ret155.1: {}
    aten::add_11:
      type: Add
      input:
        ret155.1: {}
        '485': {}
      output:
        ret156.1: {}
    aten::rsqrt_4553:
      type: Rsqrt
      input:
        ret156.1: {}
      output:
        ret157.1: {}
    aten::mul_4552:
      type: Mul
      input:
        ret153.1: {}
        ret157.1: {}
      output:
        ret158.1: {}
      attr:
        algorithm: mul
    aten::mul_354:
      type: Mul
      input:
        '516': {}
        ret158.1: {}
      output:
        ret159.1: {}
      attr:
        algorithm: mul
    aten::mul_355:
      type: Mul
      input:
        ret159.1: {}
        '517': {}
      output:
        ret160.1: {}
      attr:
        algorithm: mul
    aten::linear_3447:
      type: InnerProduct
      input:
        ret160.1: {}
        '518': {}
        aten::linear_3447_bias: {}
      output:
        ret163.1: {}
    aten::silu_4555:
      type: Swish
      input:
        ret163.1: {}
      output:
        ret164.1: {}
    aten::mul_357:
      type: Mul
      input:
        ret159.1: {}
        '519': {}
      output:
        ret165.1: {}
      attr:
        algorithm: mul
    aten::linear_3448:
      type: InnerProduct
      input:
        ret165.1: {}
        '520': {}
        aten::linear_3448_bias: {}
      output:
        ret168.1: {}
    aten::mul_4556:
      type: Mul
      input:
        ret164.1: {}
        ret168.1: {}
      output:
        ret169.1: {}
      attr:
        algorithm: mul
    aten::mul_359:
      type: Mul
      input:
        ret169.1: {}
        '521': {}
      output:
        ret170.1: {}
      attr:
        algorithm: mul
    aten::linear_3449:
      type: InnerProduct
      input:
        ret170.1: {}
        '522': {}
        aten::linear_3449_bias: {}
      output:
        ret173.1: {}
    aten::add_3080:
      type: Add
      input:
        ret153.1: {}
        ret173.1: {}
      output:
        ret174.1: {}
    aten::pow_2564:
      type: Pow
      input:
        ret174.1: {}
        aten::pow_2564_other: {}
      output:
        ret175.1: {}
    aten::mean_977:
      type: ReduceMean
      input:
        ret175.1: {}
      output:
        ret176.1: {}
    aten::add_12:
      type: Add
      input:
        ret176.1: {}
        '485': {}
      output:
        ret177.1: {}
    aten::rsqrt_4560:
      type: Rsqrt
      input:
        ret177.1: {}
      output:
        ret178.1: {}
    aten::mul_4559:
      type: Mul
      input:
        ret174.1: {}
        ret178.1: {}
      output:
        ret179.1: {}
      attr:
        algorithm: mul
    aten::mul_361:
      type: Mul
      input:
        '523': {}
        ret179.1: {}
      output:
        ret180.1: {}
      attr:
        algorithm: mul
    aten::size_3391:
      type: Shape
      input:
        ret180.1: {}
      output:
        '1858': {}
      attr:
        start: 0
        end: 0
    aten::size_3081:
      type: Shape
      input:
        ret180.1: {}
      output:
        '1860': {}
      attr:
        start: 1
        end: 1
    aten::mul_362:
      type: Mul
      input:
        ret180.1: {}
        '524': {}
      output:
        ret183.1: {}
      attr:
        algorithm: mul
    aten::linear_3450:
      type: InnerProduct
      input:
        ret183.1: {}
        '525': {}
        aten::linear_3450_bias: {}
      output:
        ret186.1: {}
    aten::view_4562:
      type: View
      input:
        ret186.1: {}
        '1858': {}
        '1860': {}
      output:
        ret187.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2565:
      type: Reorder
      input:
        ret187.1: {}
      output:
        ret188.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_364:
      type: Mul
      input:
        ret180.1: {}
        '526': {}
      output:
        ret189.1: {}
      attr:
        algorithm: mul
    aten::linear_3451:
      type: InnerProduct
      input:
        ret189.1: {}
        '527': {}
        aten::linear_3451_bias: {}
      output:
        ret192.1: {}
    aten::view_4563:
      type: View
      input:
        ret192.1: {}
        '1858': {}
        '1860': {}
      output:
        ret193.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2566:
      type: Reorder
      input:
        ret193.1: {}
      output:
        ret194.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_366:
      type: Mul
      input:
        ret180.1: {}
        '528': {}
      output:
        ret195.1: {}
      attr:
        algorithm: mul
    aten::linear_3452:
      type: InnerProduct
      input:
        ret195.1: {}
        '529': {}
        aten::linear_3452_bias: {}
      output:
        ret198.1: {}
    aten::view_4564:
      type: View
      input:
        ret198.1: {}
        '1858': {}
        '1860': {}
      output:
        ret199.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2567:
      type: Reorder
      input:
        ret199.1: {}
      output:
        ret200.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2568:
      type: Shape
      input:
        ret194.1: {}
      output:
        '1906': {}
      attr:
        start: 2
        end: 2
    aten::size_2569:
      type: Shape
      input:
        x3.1: {}
      output:
        '1909': {}
      attr:
        start: 2
        end: 2
    aten::add_3349:
      type: Add
      input:
        '1906': {}
        '1909': {}
      output:
        seq_len1.1: {}
    aten::slice_98:
      type: Slice
      input:
        '493': {}
        seq_len1.1: {}
      output:
        '1916': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_138:
      type: Slice
      input:
        '494': {}
        seq_len1.1: {}
      output:
        '1918': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2570:
      type: Shape
      input:
        ret188.1: {}
      output:
        '1920': {}
      attr:
        start: 2
        end: 2
    aten::add_3082:
      type: Add
      input:
        '1920': {}
        '1909': {}
      output:
        '1925': {}
    aten::slice_2571:
      type: Slice
      input:
        '1916': {}
        '1909': {}
        '1925': {}
      output:
        '1927': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2063:
      type: Slice
      input:
        '1927': {}
      output:
        '1928': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2572:
      type: Slice
      input:
        '1918': {}
        '1909': {}
        '1925': {}
      output:
        '1929': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2064:
      type: Slice
      input:
        '1929': {}
      output:
        '1930': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4565:
      type: Mul
      input:
        ret188.1: {}
        '1928': {}
      output:
        ret204.1: {}
      attr:
        algorithm: mul
    aten::size_2065:
      type: Shape
      input:
        ret188.1: {}
      output:
        '1934': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_180:
      type: Div
      input:
        '1934': {}
        '495': {}
      output:
        ret206.1: {}
      attr:
        algorithm: div
    aten::slice_2066:
      type: Slice
      input:
        ret188.1: {}
        ret206.1: {}
      output:
        ret207.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2067:
      type: Slice
      input:
        ret188.1: {}
        ret206.1: {}
      output:
        ret208.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4572:
      type: Neg
      input:
        ret208.1: {}
        aten::neg_4572_mul_val: {}
      output:
        ret209.1: {}
      attr:
        algorithm: mul
    aten::cat_2457:
      type: Concat
      input:
        ret209.1: {}
        ret207.1: {}
      output:
        ret210.1: {}
      attr:
        axis: -1
    aten::mul_4569:
      type: Mul
      input:
        ret210.1: {}
        '1930': {}
      output:
        ret211.1: {}
      attr:
        algorithm: mul
    aten::add_3083:
      type: Add
      input:
        ret204.1: {}
        ret211.1: {}
      output:
        args7.1: {}
    aten::mul_4567:
      type: Mul
      input:
        ret194.1: {}
        '1928': {}
      output:
        ret212.1: {}
      attr:
        algorithm: mul
    aten::size_2068:
      type: Shape
      input:
        ret194.1: {}
      output:
        '1956': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_181:
      type: Div
      input:
        '1956': {}
        '495': {}
      output:
        ret214.1: {}
      attr:
        algorithm: div
    aten::slice_2069:
      type: Slice
      input:
        ret194.1: {}
        ret214.1: {}
      output:
        ret215.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2070:
      type: Slice
      input:
        ret194.1: {}
        ret214.1: {}
      output:
        ret216.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4574:
      type: Neg
      input:
        ret216.1: {}
        aten::neg_4574_mul_val: {}
      output:
        ret217.1: {}
      attr:
        algorithm: mul
    aten::cat_2458:
      type: Concat
      input:
        ret217.1: {}
        ret215.1: {}
      output:
        ret218.1: {}
      attr:
        axis: -1
    aten::mul_4570:
      type: Mul
      input:
        ret218.1: {}
        '1930': {}
      output:
        ret219.1: {}
      attr:
        algorithm: mul
    aten::add_3084:
      type: Add
      input:
        ret212.1: {}
        ret219.1: {}
      output:
        '1974': {}
    aten::cat_2573:
      type: Concat
      input:
        x3.1: {}
        '1974': {}
      output:
        ret220.1: {}
      attr:
        axis: 2
    aten::cat_2574:
      type: Concat
      input:
        x4.1: {}
        ret200.1: {}
      output:
        ret221.1: {}
      attr:
        axis: 2
    aten::transpose_2071:
      type: Reorder
      input:
        ret220.1: {}
      output:
        ret222.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4577:
      type: Matmul
      input:
        args7.1: {}
        ret222.1: {}
      output:
        ret225.1: {}
    aten::div_258:
      type: Div
      input:
        ret225.1: {}
        '496': {}
      output:
        ret226.1: {}
      attr:
        algorithm: div
    aten::add_3085:
      type: Add
      input:
        ret226.1: {}
        attention_mask0.1: {}
      output:
        attn_weights4.1: {}
    aten::max_298:
      type: Max
      input:
        attn_weights4.1: {}
        '497': {}
      output:
        input4.1: {}
    aten::softmax_2411:
      type: Softmax
      input:
        input4.1: {}
      output:
        '1996': {}
      attr:
        axis: -1
    aten::matmul_4580:
      type: Matmul
      input:
        '1996': {}
        ret221.1: {}
      output:
        ret229.1: {}
    aten::transpose_2575:
      type: Reorder
      input:
        ret229.1: {}
      output:
        ret230.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4582:
      type: Reshape
      input:
        ret230.1: {}
        '1858': {}
        '1860': {}
      output:
        ret231.1: {}
    aten::mul_368:
      type: Mul
      input:
        ret231.1: {}
        '530': {}
      output:
        ret232.1: {}
      attr:
        algorithm: mul
    aten::linear_3453:
      type: InnerProduct
      input:
        ret232.1: {}
        '531': {}
        aten::linear_3453_bias: {}
      output:
        ret235.1: {}
    aten::add_3086:
      type: Add
      input:
        ret174.1: {}
        ret235.1: {}
      output:
        ret236.1: {}
    aten::pow_2576:
      type: Pow
      input:
        ret236.1: {}
        aten::pow_2576_other: {}
      output:
        ret237.1: {}
    aten::mean_978:
      type: ReduceMean
      input:
        ret237.1: {}
      output:
        ret238.1: {}
    aten::add_13:
      type: Add
      input:
        ret238.1: {}
        '485': {}
      output:
        ret239.1: {}
    aten::rsqrt_4585:
      type: Rsqrt
      input:
        ret239.1: {}
      output:
        ret240.1: {}
    aten::mul_4584:
      type: Mul
      input:
        ret236.1: {}
        ret240.1: {}
      output:
        ret241.1: {}
      attr:
        algorithm: mul
    aten::mul_370:
      type: Mul
      input:
        '532': {}
        ret241.1: {}
      output:
        ret242.1: {}
      attr:
        algorithm: mul
    aten::mul_371:
      type: Mul
      input:
        ret242.1: {}
        '533': {}
      output:
        ret243.1: {}
      attr:
        algorithm: mul
    aten::linear_3454:
      type: InnerProduct
      input:
        ret243.1: {}
        '534': {}
        aten::linear_3454_bias: {}
      output:
        ret246.1: {}
    aten::silu_4587:
      type: Swish
      input:
        ret246.1: {}
      output:
        ret247.1: {}
    aten::mul_373:
      type: Mul
      input:
        ret242.1: {}
        '535': {}
      output:
        ret248.1: {}
      attr:
        algorithm: mul
    aten::linear_3455:
      type: InnerProduct
      input:
        ret248.1: {}
        '536': {}
        aten::linear_3455_bias: {}
      output:
        ret251.1: {}
    aten::mul_4588:
      type: Mul
      input:
        ret247.1: {}
        ret251.1: {}
      output:
        ret252.1: {}
      attr:
        algorithm: mul
    aten::mul_375:
      type: Mul
      input:
        ret252.1: {}
        '537': {}
      output:
        ret253.1: {}
      attr:
        algorithm: mul
    aten::linear_3456:
      type: InnerProduct
      input:
        ret253.1: {}
        '538': {}
        aten::linear_3456_bias: {}
      output:
        ret256.1: {}
    aten::add_3087:
      type: Add
      input:
        ret236.1: {}
        ret256.1: {}
      output:
        ret257.1: {}
    aten::pow_2577:
      type: Pow
      input:
        ret257.1: {}
        aten::pow_2577_other: {}
      output:
        ret258.1: {}
    aten::mean_979:
      type: ReduceMean
      input:
        ret258.1: {}
      output:
        ret259.1: {}
    aten::add_14:
      type: Add
      input:
        ret259.1: {}
        '485': {}
      output:
        ret260.1: {}
    aten::rsqrt_4592:
      type: Rsqrt
      input:
        ret260.1: {}
      output:
        ret261.1: {}
    aten::mul_4591:
      type: Mul
      input:
        ret257.1: {}
        ret261.1: {}
      output:
        ret262.1: {}
      attr:
        algorithm: mul
    aten::mul_377:
      type: Mul
      input:
        '539': {}
        ret262.1: {}
      output:
        ret263.1: {}
      attr:
        algorithm: mul
    aten::size_3392:
      type: Shape
      input:
        ret263.1: {}
      output:
        '2081': {}
      attr:
        start: 0
        end: 0
    aten::size_3088:
      type: Shape
      input:
        ret263.1: {}
      output:
        '2083': {}
      attr:
        start: 1
        end: 1
    aten::mul_378:
      type: Mul
      input:
        ret263.1: {}
        '540': {}
      output:
        ret266.1: {}
      attr:
        algorithm: mul
    aten::linear_3457:
      type: InnerProduct
      input:
        ret266.1: {}
        '541': {}
        aten::linear_3457_bias: {}
      output:
        ret269.1: {}
    aten::view_4594:
      type: View
      input:
        ret269.1: {}
        '2081': {}
        '2083': {}
      output:
        ret270.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2578:
      type: Reorder
      input:
        ret270.1: {}
      output:
        ret271.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_380:
      type: Mul
      input:
        ret263.1: {}
        '542': {}
      output:
        ret272.1: {}
      attr:
        algorithm: mul
    aten::linear_3458:
      type: InnerProduct
      input:
        ret272.1: {}
        '543': {}
        aten::linear_3458_bias: {}
      output:
        ret275.1: {}
    aten::view_4595:
      type: View
      input:
        ret275.1: {}
        '2081': {}
        '2083': {}
      output:
        ret276.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2579:
      type: Reorder
      input:
        ret276.1: {}
      output:
        ret277.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_382:
      type: Mul
      input:
        ret263.1: {}
        '544': {}
      output:
        ret278.1: {}
      attr:
        algorithm: mul
    aten::linear_3459:
      type: InnerProduct
      input:
        ret278.1: {}
        '545': {}
        aten::linear_3459_bias: {}
      output:
        ret281.1: {}
    aten::view_4596:
      type: View
      input:
        ret281.1: {}
        '2081': {}
        '2083': {}
      output:
        ret282.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2580:
      type: Reorder
      input:
        ret282.1: {}
      output:
        ret283.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2581:
      type: Shape
      input:
        ret277.1: {}
      output:
        '2129': {}
      attr:
        start: 2
        end: 2
    aten::size_2582:
      type: Shape
      input:
        x5.1: {}
      output:
        '2132': {}
      attr:
        start: 2
        end: 2
    aten::add_3350:
      type: Add
      input:
        '2129': {}
        '2132': {}
      output:
        seq_len2.1: {}
    aten::slice_99:
      type: Slice
      input:
        '493': {}
        seq_len2.1: {}
      output:
        '2139': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_139:
      type: Slice
      input:
        '494': {}
        seq_len2.1: {}
      output:
        '2141': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2583:
      type: Shape
      input:
        ret271.1: {}
      output:
        '2143': {}
      attr:
        start: 2
        end: 2
    aten::add_3089:
      type: Add
      input:
        '2143': {}
        '2132': {}
      output:
        '2148': {}
    aten::slice_2584:
      type: Slice
      input:
        '2139': {}
        '2132': {}
        '2148': {}
      output:
        '2150': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2072:
      type: Slice
      input:
        '2150': {}
      output:
        '2151': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2585:
      type: Slice
      input:
        '2141': {}
        '2132': {}
        '2148': {}
      output:
        '2152': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2073:
      type: Slice
      input:
        '2152': {}
      output:
        '2153': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4597:
      type: Mul
      input:
        ret271.1: {}
        '2151': {}
      output:
        ret287.1: {}
      attr:
        algorithm: mul
    aten::size_2074:
      type: Shape
      input:
        ret271.1: {}
      output:
        '2157': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_182:
      type: Div
      input:
        '2157': {}
        '495': {}
      output:
        ret289.1: {}
      attr:
        algorithm: div
    aten::slice_2075:
      type: Slice
      input:
        ret271.1: {}
        ret289.1: {}
      output:
        ret290.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2076:
      type: Slice
      input:
        ret271.1: {}
        ret289.1: {}
      output:
        ret291.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4604:
      type: Neg
      input:
        ret291.1: {}
        aten::neg_4604_mul_val: {}
      output:
        ret292.1: {}
      attr:
        algorithm: mul
    aten::cat_2459:
      type: Concat
      input:
        ret292.1: {}
        ret290.1: {}
      output:
        ret293.1: {}
      attr:
        axis: -1
    aten::mul_4601:
      type: Mul
      input:
        ret293.1: {}
        '2153': {}
      output:
        ret294.1: {}
      attr:
        algorithm: mul
    aten::add_3090:
      type: Add
      input:
        ret287.1: {}
        ret294.1: {}
      output:
        args11.1: {}
    aten::mul_4599:
      type: Mul
      input:
        ret277.1: {}
        '2151': {}
      output:
        ret295.1: {}
      attr:
        algorithm: mul
    aten::size_2077:
      type: Shape
      input:
        ret277.1: {}
      output:
        '2179': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_183:
      type: Div
      input:
        '2179': {}
        '495': {}
      output:
        ret297.1: {}
      attr:
        algorithm: div
    aten::slice_2078:
      type: Slice
      input:
        ret277.1: {}
        ret297.1: {}
      output:
        ret298.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2079:
      type: Slice
      input:
        ret277.1: {}
        ret297.1: {}
      output:
        ret299.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4606:
      type: Neg
      input:
        ret299.1: {}
        aten::neg_4606_mul_val: {}
      output:
        ret300.1: {}
      attr:
        algorithm: mul
    aten::cat_2460:
      type: Concat
      input:
        ret300.1: {}
        ret298.1: {}
      output:
        ret301.1: {}
      attr:
        axis: -1
    aten::mul_4602:
      type: Mul
      input:
        ret301.1: {}
        '2153': {}
      output:
        ret302.1: {}
      attr:
        algorithm: mul
    aten::add_3091:
      type: Add
      input:
        ret295.1: {}
        ret302.1: {}
      output:
        '2197': {}
    aten::cat_2586:
      type: Concat
      input:
        x5.1: {}
        '2197': {}
      output:
        ret303.1: {}
      attr:
        axis: 2
    aten::cat_2587:
      type: Concat
      input:
        x6.1: {}
        ret283.1: {}
      output:
        ret304.1: {}
      attr:
        axis: 2
    aten::transpose_2080:
      type: Reorder
      input:
        ret303.1: {}
      output:
        ret305.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4609:
      type: Matmul
      input:
        args11.1: {}
        ret305.1: {}
      output:
        ret308.1: {}
    aten::div_259:
      type: Div
      input:
        ret308.1: {}
        '496': {}
      output:
        ret309.1: {}
      attr:
        algorithm: div
    aten::add_3092:
      type: Add
      input:
        ret309.1: {}
        attention_mask0.1: {}
      output:
        attn_weights6.1: {}
    aten::max_299:
      type: Max
      input:
        attn_weights6.1: {}
        '497': {}
      output:
        input6.1: {}
    aten::softmax_2412:
      type: Softmax
      input:
        input6.1: {}
      output:
        '2219': {}
      attr:
        axis: -1
    aten::matmul_4612:
      type: Matmul
      input:
        '2219': {}
        ret304.1: {}
      output:
        ret312.1: {}
    aten::transpose_2588:
      type: Reorder
      input:
        ret312.1: {}
      output:
        ret313.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4614:
      type: Reshape
      input:
        ret313.1: {}
        '2081': {}
        '2083': {}
      output:
        ret314.1: {}
    aten::mul_384:
      type: Mul
      input:
        ret314.1: {}
        '546': {}
      output:
        ret315.1: {}
      attr:
        algorithm: mul
    aten::linear_3460:
      type: InnerProduct
      input:
        ret315.1: {}
        '547': {}
        aten::linear_3460_bias: {}
      output:
        ret318.1: {}
    aten::add_3093:
      type: Add
      input:
        ret257.1: {}
        ret318.1: {}
      output:
        ret319.1: {}
    aten::pow_2589:
      type: Pow
      input:
        ret319.1: {}
        aten::pow_2589_other: {}
      output:
        ret320.1: {}
    aten::mean_980:
      type: ReduceMean
      input:
        ret320.1: {}
      output:
        ret321.1: {}
    aten::add_15:
      type: Add
      input:
        ret321.1: {}
        '485': {}
      output:
        ret322.1: {}
    aten::rsqrt_4617:
      type: Rsqrt
      input:
        ret322.1: {}
      output:
        ret323.1: {}
    aten::mul_4616:
      type: Mul
      input:
        ret319.1: {}
        ret323.1: {}
      output:
        ret324.1: {}
      attr:
        algorithm: mul
    aten::mul_386:
      type: Mul
      input:
        '548': {}
        ret324.1: {}
      output:
        ret325.1: {}
      attr:
        algorithm: mul
    aten::mul_387:
      type: Mul
      input:
        ret325.1: {}
        '549': {}
      output:
        ret326.1: {}
      attr:
        algorithm: mul
    aten::linear_3461:
      type: InnerProduct
      input:
        ret326.1: {}
        '550': {}
        aten::linear_3461_bias: {}
      output:
        ret329.1: {}
    aten::silu_4619:
      type: Swish
      input:
        ret329.1: {}
      output:
        ret330.1: {}
    aten::mul_389:
      type: Mul
      input:
        ret325.1: {}
        '551': {}
      output:
        ret331.1: {}
      attr:
        algorithm: mul
    aten::linear_3462:
      type: InnerProduct
      input:
        ret331.1: {}
        '552': {}
        aten::linear_3462_bias: {}
      output:
        ret334.1: {}
    aten::mul_4620:
      type: Mul
      input:
        ret330.1: {}
        ret334.1: {}
      output:
        ret335.1: {}
      attr:
        algorithm: mul
    aten::mul_391:
      type: Mul
      input:
        ret335.1: {}
        '553': {}
      output:
        ret336.1: {}
      attr:
        algorithm: mul
    aten::linear_3463:
      type: InnerProduct
      input:
        ret336.1: {}
        '554': {}
        aten::linear_3463_bias: {}
      output:
        ret339.1: {}
    aten::add_3094:
      type: Add
      input:
        ret319.1: {}
        ret339.1: {}
      output:
        ret340.1: {}
    aten::pow_2590:
      type: Pow
      input:
        ret340.1: {}
        aten::pow_2590_other: {}
      output:
        ret341.1: {}
    aten::mean_981:
      type: ReduceMean
      input:
        ret341.1: {}
      output:
        ret342.1: {}
    aten::add_16:
      type: Add
      input:
        ret342.1: {}
        '485': {}
      output:
        ret343.1: {}
    aten::rsqrt_4624:
      type: Rsqrt
      input:
        ret343.1: {}
      output:
        ret344.1: {}
    aten::mul_4623:
      type: Mul
      input:
        ret340.1: {}
        ret344.1: {}
      output:
        ret345.1: {}
      attr:
        algorithm: mul
    aten::mul_393:
      type: Mul
      input:
        '555': {}
        ret345.1: {}
      output:
        ret346.1: {}
      attr:
        algorithm: mul
    aten::size_3393:
      type: Shape
      input:
        ret346.1: {}
      output:
        '2304': {}
      attr:
        start: 0
        end: 0
    aten::size_3095:
      type: Shape
      input:
        ret346.1: {}
      output:
        '2306': {}
      attr:
        start: 1
        end: 1
    aten::mul_394:
      type: Mul
      input:
        ret346.1: {}
        '556': {}
      output:
        ret349.1: {}
      attr:
        algorithm: mul
    aten::linear_3464:
      type: InnerProduct
      input:
        ret349.1: {}
        '557': {}
        aten::linear_3464_bias: {}
      output:
        ret352.1: {}
    aten::view_4626:
      type: View
      input:
        ret352.1: {}
        '2304': {}
        '2306': {}
      output:
        ret353.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2591:
      type: Reorder
      input:
        ret353.1: {}
      output:
        ret354.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_396:
      type: Mul
      input:
        ret346.1: {}
        '558': {}
      output:
        ret355.1: {}
      attr:
        algorithm: mul
    aten::linear_3465:
      type: InnerProduct
      input:
        ret355.1: {}
        '559': {}
        aten::linear_3465_bias: {}
      output:
        ret358.1: {}
    aten::view_4627:
      type: View
      input:
        ret358.1: {}
        '2304': {}
        '2306': {}
      output:
        ret359.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2592:
      type: Reorder
      input:
        ret359.1: {}
      output:
        ret360.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_398:
      type: Mul
      input:
        ret346.1: {}
        '560': {}
      output:
        ret361.1: {}
      attr:
        algorithm: mul
    aten::linear_3466:
      type: InnerProduct
      input:
        ret361.1: {}
        '561': {}
        aten::linear_3466_bias: {}
      output:
        ret364.1: {}
    aten::view_4628:
      type: View
      input:
        ret364.1: {}
        '2304': {}
        '2306': {}
      output:
        ret365.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2593:
      type: Reorder
      input:
        ret365.1: {}
      output:
        ret366.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2594:
      type: Shape
      input:
        ret360.1: {}
      output:
        '2352': {}
      attr:
        start: 2
        end: 2
    aten::size_2595:
      type: Shape
      input:
        x7.1: {}
      output:
        '2355': {}
      attr:
        start: 2
        end: 2
    aten::add_3351:
      type: Add
      input:
        '2352': {}
        '2355': {}
      output:
        seq_len3.1: {}
    aten::slice_100:
      type: Slice
      input:
        '493': {}
        seq_len3.1: {}
      output:
        '2362': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_140:
      type: Slice
      input:
        '494': {}
        seq_len3.1: {}
      output:
        '2364': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2596:
      type: Shape
      input:
        ret354.1: {}
      output:
        '2366': {}
      attr:
        start: 2
        end: 2
    aten::add_3096:
      type: Add
      input:
        '2366': {}
        '2355': {}
      output:
        '2371': {}
    aten::slice_2597:
      type: Slice
      input:
        '2362': {}
        '2355': {}
        '2371': {}
      output:
        '2373': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2081:
      type: Slice
      input:
        '2373': {}
      output:
        '2374': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2598:
      type: Slice
      input:
        '2364': {}
        '2355': {}
        '2371': {}
      output:
        '2375': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2082:
      type: Slice
      input:
        '2375': {}
      output:
        '2376': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4629:
      type: Mul
      input:
        ret354.1: {}
        '2374': {}
      output:
        ret370.1: {}
      attr:
        algorithm: mul
    aten::size_2083:
      type: Shape
      input:
        ret354.1: {}
      output:
        '2380': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_184:
      type: Div
      input:
        '2380': {}
        '495': {}
      output:
        ret372.1: {}
      attr:
        algorithm: div
    aten::slice_2084:
      type: Slice
      input:
        ret354.1: {}
        ret372.1: {}
      output:
        ret373.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2085:
      type: Slice
      input:
        ret354.1: {}
        ret372.1: {}
      output:
        ret374.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4636:
      type: Neg
      input:
        ret374.1: {}
        aten::neg_4636_mul_val: {}
      output:
        ret375.1: {}
      attr:
        algorithm: mul
    aten::cat_2461:
      type: Concat
      input:
        ret375.1: {}
        ret373.1: {}
      output:
        ret376.1: {}
      attr:
        axis: -1
    aten::mul_4633:
      type: Mul
      input:
        ret376.1: {}
        '2376': {}
      output:
        ret377.1: {}
      attr:
        algorithm: mul
    aten::add_3097:
      type: Add
      input:
        ret370.1: {}
        ret377.1: {}
      output:
        args15.1: {}
    aten::mul_4631:
      type: Mul
      input:
        ret360.1: {}
        '2374': {}
      output:
        ret378.1: {}
      attr:
        algorithm: mul
    aten::size_2086:
      type: Shape
      input:
        ret360.1: {}
      output:
        '2402': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_185:
      type: Div
      input:
        '2402': {}
        '495': {}
      output:
        ret380.1: {}
      attr:
        algorithm: div
    aten::slice_2087:
      type: Slice
      input:
        ret360.1: {}
        ret380.1: {}
      output:
        ret381.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2088:
      type: Slice
      input:
        ret360.1: {}
        ret380.1: {}
      output:
        ret382.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4638:
      type: Neg
      input:
        ret382.1: {}
        aten::neg_4638_mul_val: {}
      output:
        ret383.1: {}
      attr:
        algorithm: mul
    aten::cat_2462:
      type: Concat
      input:
        ret383.1: {}
        ret381.1: {}
      output:
        ret384.1: {}
      attr:
        axis: -1
    aten::mul_4634:
      type: Mul
      input:
        ret384.1: {}
        '2376': {}
      output:
        ret385.1: {}
      attr:
        algorithm: mul
    aten::add_3098:
      type: Add
      input:
        ret378.1: {}
        ret385.1: {}
      output:
        '2420': {}
    aten::cat_2599:
      type: Concat
      input:
        x7.1: {}
        '2420': {}
      output:
        ret386.1: {}
      attr:
        axis: 2
    aten::cat_2600:
      type: Concat
      input:
        x8.1: {}
        ret366.1: {}
      output:
        ret387.1: {}
      attr:
        axis: 2
    aten::transpose_2089:
      type: Reorder
      input:
        ret386.1: {}
      output:
        ret388.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4641:
      type: Matmul
      input:
        args15.1: {}
        ret388.1: {}
      output:
        ret391.1: {}
    aten::div_260:
      type: Div
      input:
        ret391.1: {}
        '496': {}
      output:
        ret392.1: {}
      attr:
        algorithm: div
    aten::add_3099:
      type: Add
      input:
        ret392.1: {}
        attention_mask0.1: {}
      output:
        attn_weights8.1: {}
    aten::max_300:
      type: Max
      input:
        attn_weights8.1: {}
        '497': {}
      output:
        input8.1: {}
    aten::softmax_2413:
      type: Softmax
      input:
        input8.1: {}
      output:
        '2442': {}
      attr:
        axis: -1
    aten::matmul_4644:
      type: Matmul
      input:
        '2442': {}
        ret387.1: {}
      output:
        ret395.1: {}
    aten::transpose_2601:
      type: Reorder
      input:
        ret395.1: {}
      output:
        ret396.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4646:
      type: Reshape
      input:
        ret396.1: {}
        '2304': {}
        '2306': {}
      output:
        ret397.1: {}
    aten::mul_400:
      type: Mul
      input:
        ret397.1: {}
        '562': {}
      output:
        ret398.1: {}
      attr:
        algorithm: mul
    aten::linear_3467:
      type: InnerProduct
      input:
        ret398.1: {}
        '563': {}
        aten::linear_3467_bias: {}
      output:
        ret401.1: {}
    aten::add_3100:
      type: Add
      input:
        ret340.1: {}
        ret401.1: {}
      output:
        ret402.1: {}
    aten::pow_2602:
      type: Pow
      input:
        ret402.1: {}
        aten::pow_2602_other: {}
      output:
        ret403.1: {}
    aten::mean_982:
      type: ReduceMean
      input:
        ret403.1: {}
      output:
        ret404.1: {}
    aten::add_17:
      type: Add
      input:
        ret404.1: {}
        '485': {}
      output:
        ret405.1: {}
    aten::rsqrt_4649:
      type: Rsqrt
      input:
        ret405.1: {}
      output:
        ret406.1: {}
    aten::mul_4648:
      type: Mul
      input:
        ret402.1: {}
        ret406.1: {}
      output:
        ret407.1: {}
      attr:
        algorithm: mul
    aten::mul_402:
      type: Mul
      input:
        '564': {}
        ret407.1: {}
      output:
        ret408.1: {}
      attr:
        algorithm: mul
    aten::mul_403:
      type: Mul
      input:
        ret408.1: {}
        '565': {}
      output:
        ret409.1: {}
      attr:
        algorithm: mul
    aten::linear_3468:
      type: InnerProduct
      input:
        ret409.1: {}
        '566': {}
        aten::linear_3468_bias: {}
      output:
        ret412.1: {}
    aten::silu_4651:
      type: Swish
      input:
        ret412.1: {}
      output:
        ret413.1: {}
    aten::mul_405:
      type: Mul
      input:
        ret408.1: {}
        '567': {}
      output:
        ret414.1: {}
      attr:
        algorithm: mul
    aten::linear_3469:
      type: InnerProduct
      input:
        ret414.1: {}
        '568': {}
        aten::linear_3469_bias: {}
      output:
        ret417.1: {}
    aten::mul_4652:
      type: Mul
      input:
        ret413.1: {}
        ret417.1: {}
      output:
        ret418.1: {}
      attr:
        algorithm: mul
    aten::mul_407:
      type: Mul
      input:
        ret418.1: {}
        '569': {}
      output:
        ret419.1: {}
      attr:
        algorithm: mul
    aten::linear_3470:
      type: InnerProduct
      input:
        ret419.1: {}
        '570': {}
        aten::linear_3470_bias: {}
      output:
        ret422.1: {}
    aten::add_3101:
      type: Add
      input:
        ret402.1: {}
        ret422.1: {}
      output:
        ret423.1: {}
    aten::pow_2603:
      type: Pow
      input:
        ret423.1: {}
        aten::pow_2603_other: {}
      output:
        ret424.1: {}
    aten::mean_983:
      type: ReduceMean
      input:
        ret424.1: {}
      output:
        ret425.1: {}
    aten::add_18:
      type: Add
      input:
        ret425.1: {}
        '485': {}
      output:
        ret426.1: {}
    aten::rsqrt_4656:
      type: Rsqrt
      input:
        ret426.1: {}
      output:
        ret427.1: {}
    aten::mul_4655:
      type: Mul
      input:
        ret423.1: {}
        ret427.1: {}
      output:
        ret428.1: {}
      attr:
        algorithm: mul
    aten::mul_409:
      type: Mul
      input:
        '571': {}
        ret428.1: {}
      output:
        ret429.1: {}
      attr:
        algorithm: mul
    aten::size_3394:
      type: Shape
      input:
        ret429.1: {}
      output:
        '2527': {}
      attr:
        start: 0
        end: 0
    aten::size_3102:
      type: Shape
      input:
        ret429.1: {}
      output:
        '2529': {}
      attr:
        start: 1
        end: 1
    aten::mul_410:
      type: Mul
      input:
        ret429.1: {}
        '572': {}
      output:
        ret432.1: {}
      attr:
        algorithm: mul
    aten::linear_3471:
      type: InnerProduct
      input:
        ret432.1: {}
        '573': {}
        aten::linear_3471_bias: {}
      output:
        ret435.1: {}
    aten::view_4658:
      type: View
      input:
        ret435.1: {}
        '2527': {}
        '2529': {}
      output:
        ret436.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2604:
      type: Reorder
      input:
        ret436.1: {}
      output:
        ret437.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_412:
      type: Mul
      input:
        ret429.1: {}
        '574': {}
      output:
        ret438.1: {}
      attr:
        algorithm: mul
    aten::linear_3472:
      type: InnerProduct
      input:
        ret438.1: {}
        '575': {}
        aten::linear_3472_bias: {}
      output:
        ret441.1: {}
    aten::view_4659:
      type: View
      input:
        ret441.1: {}
        '2527': {}
        '2529': {}
      output:
        ret442.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2605:
      type: Reorder
      input:
        ret442.1: {}
      output:
        ret443.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_414:
      type: Mul
      input:
        ret429.1: {}
        '576': {}
      output:
        ret444.1: {}
      attr:
        algorithm: mul
    aten::linear_3473:
      type: InnerProduct
      input:
        ret444.1: {}
        '577': {}
        aten::linear_3473_bias: {}
      output:
        ret447.1: {}
    aten::view_4660:
      type: View
      input:
        ret447.1: {}
        '2527': {}
        '2529': {}
      output:
        ret448.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2606:
      type: Reorder
      input:
        ret448.1: {}
      output:
        ret449.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2607:
      type: Shape
      input:
        ret443.1: {}
      output:
        '2575': {}
      attr:
        start: 2
        end: 2
    aten::size_2608:
      type: Shape
      input:
        x9.1: {}
      output:
        '2578': {}
      attr:
        start: 2
        end: 2
    aten::add_3352:
      type: Add
      input:
        '2575': {}
        '2578': {}
      output:
        seq_len4.1: {}
    aten::slice_101:
      type: Slice
      input:
        '493': {}
        seq_len4.1: {}
      output:
        '2585': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_141:
      type: Slice
      input:
        '494': {}
        seq_len4.1: {}
      output:
        '2587': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2609:
      type: Shape
      input:
        ret437.1: {}
      output:
        '2589': {}
      attr:
        start: 2
        end: 2
    aten::add_3103:
      type: Add
      input:
        '2589': {}
        '2578': {}
      output:
        '2594': {}
    aten::slice_2610:
      type: Slice
      input:
        '2585': {}
        '2578': {}
        '2594': {}
      output:
        '2596': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2090:
      type: Slice
      input:
        '2596': {}
      output:
        '2597': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2611:
      type: Slice
      input:
        '2587': {}
        '2578': {}
        '2594': {}
      output:
        '2598': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2091:
      type: Slice
      input:
        '2598': {}
      output:
        '2599': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4661:
      type: Mul
      input:
        ret437.1: {}
        '2597': {}
      output:
        ret453.1: {}
      attr:
        algorithm: mul
    aten::size_2092:
      type: Shape
      input:
        ret437.1: {}
      output:
        '2603': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_186:
      type: Div
      input:
        '2603': {}
        '495': {}
      output:
        ret455.1: {}
      attr:
        algorithm: div
    aten::slice_2093:
      type: Slice
      input:
        ret437.1: {}
        ret455.1: {}
      output:
        ret456.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2094:
      type: Slice
      input:
        ret437.1: {}
        ret455.1: {}
      output:
        ret457.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4668:
      type: Neg
      input:
        ret457.1: {}
        aten::neg_4668_mul_val: {}
      output:
        ret458.1: {}
      attr:
        algorithm: mul
    aten::cat_2463:
      type: Concat
      input:
        ret458.1: {}
        ret456.1: {}
      output:
        ret459.1: {}
      attr:
        axis: -1
    aten::mul_4665:
      type: Mul
      input:
        ret459.1: {}
        '2599': {}
      output:
        ret460.1: {}
      attr:
        algorithm: mul
    aten::add_3104:
      type: Add
      input:
        ret453.1: {}
        ret460.1: {}
      output:
        args19.1: {}
    aten::mul_4663:
      type: Mul
      input:
        ret443.1: {}
        '2597': {}
      output:
        ret461.1: {}
      attr:
        algorithm: mul
    aten::size_2095:
      type: Shape
      input:
        ret443.1: {}
      output:
        '2625': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_187:
      type: Div
      input:
        '2625': {}
        '495': {}
      output:
        ret463.1: {}
      attr:
        algorithm: div
    aten::slice_2096:
      type: Slice
      input:
        ret443.1: {}
        ret463.1: {}
      output:
        ret464.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2097:
      type: Slice
      input:
        ret443.1: {}
        ret463.1: {}
      output:
        ret465.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4670:
      type: Neg
      input:
        ret465.1: {}
        aten::neg_4670_mul_val: {}
      output:
        ret466.1: {}
      attr:
        algorithm: mul
    aten::cat_2464:
      type: Concat
      input:
        ret466.1: {}
        ret464.1: {}
      output:
        ret467.1: {}
      attr:
        axis: -1
    aten::mul_4666:
      type: Mul
      input:
        ret467.1: {}
        '2599': {}
      output:
        ret468.1: {}
      attr:
        algorithm: mul
    aten::add_3105:
      type: Add
      input:
        ret461.1: {}
        ret468.1: {}
      output:
        '2643': {}
    aten::cat_2612:
      type: Concat
      input:
        x9.1: {}
        '2643': {}
      output:
        ret469.1: {}
      attr:
        axis: 2
    aten::cat_2613:
      type: Concat
      input:
        x10.1: {}
        ret449.1: {}
      output:
        ret470.1: {}
      attr:
        axis: 2
    aten::transpose_2098:
      type: Reorder
      input:
        ret469.1: {}
      output:
        ret471.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4673:
      type: Matmul
      input:
        args19.1: {}
        ret471.1: {}
      output:
        ret474.1: {}
    aten::div_261:
      type: Div
      input:
        ret474.1: {}
        '496': {}
      output:
        ret475.1: {}
      attr:
        algorithm: div
    aten::add_3106:
      type: Add
      input:
        ret475.1: {}
        attention_mask0.1: {}
      output:
        attn_weights10.1: {}
    aten::max_301:
      type: Max
      input:
        attn_weights10.1: {}
        '497': {}
      output:
        input10.1: {}
    aten::softmax_2414:
      type: Softmax
      input:
        input10.1: {}
      output:
        '2665': {}
      attr:
        axis: -1
    aten::matmul_4676:
      type: Matmul
      input:
        '2665': {}
        ret470.1: {}
      output:
        ret478.1: {}
    aten::transpose_2614:
      type: Reorder
      input:
        ret478.1: {}
      output:
        ret479.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4678:
      type: Reshape
      input:
        ret479.1: {}
        '2527': {}
        '2529': {}
      output:
        ret480.1: {}
    aten::mul_416:
      type: Mul
      input:
        ret480.1: {}
        '578': {}
      output:
        ret481.1: {}
      attr:
        algorithm: mul
    aten::linear_3474:
      type: InnerProduct
      input:
        ret481.1: {}
        '579': {}
        aten::linear_3474_bias: {}
      output:
        ret484.1: {}
    aten::add_3107:
      type: Add
      input:
        ret423.1: {}
        ret484.1: {}
      output:
        ret485.1: {}
    aten::pow_2615:
      type: Pow
      input:
        ret485.1: {}
        aten::pow_2615_other: {}
      output:
        ret486.1: {}
    aten::mean_984:
      type: ReduceMean
      input:
        ret486.1: {}
      output:
        ret487.1: {}
    aten::add_19:
      type: Add
      input:
        ret487.1: {}
        '485': {}
      output:
        ret488.1: {}
    aten::rsqrt_4681:
      type: Rsqrt
      input:
        ret488.1: {}
      output:
        ret489.1: {}
    aten::mul_4680:
      type: Mul
      input:
        ret485.1: {}
        ret489.1: {}
      output:
        ret490.1: {}
      attr:
        algorithm: mul
    aten::mul_418:
      type: Mul
      input:
        '580': {}
        ret490.1: {}
      output:
        ret491.1: {}
      attr:
        algorithm: mul
    aten::mul_419:
      type: Mul
      input:
        ret491.1: {}
        '581': {}
      output:
        ret492.1: {}
      attr:
        algorithm: mul
    aten::linear_3475:
      type: InnerProduct
      input:
        ret492.1: {}
        '582': {}
        aten::linear_3475_bias: {}
      output:
        ret495.1: {}
    aten::silu_4683:
      type: Swish
      input:
        ret495.1: {}
      output:
        ret496.1: {}
    aten::mul_421:
      type: Mul
      input:
        ret491.1: {}
        '583': {}
      output:
        ret497.1: {}
      attr:
        algorithm: mul
    aten::linear_3476:
      type: InnerProduct
      input:
        ret497.1: {}
        '584': {}
        aten::linear_3476_bias: {}
      output:
        ret500.1: {}
    aten::mul_4684:
      type: Mul
      input:
        ret496.1: {}
        ret500.1: {}
      output:
        ret501.1: {}
      attr:
        algorithm: mul
    aten::mul_423:
      type: Mul
      input:
        ret501.1: {}
        '585': {}
      output:
        ret502.1: {}
      attr:
        algorithm: mul
    aten::linear_3477:
      type: InnerProduct
      input:
        ret502.1: {}
        '586': {}
        aten::linear_3477_bias: {}
      output:
        ret505.1: {}
    aten::add_3108:
      type: Add
      input:
        ret485.1: {}
        ret505.1: {}
      output:
        ret506.1: {}
    aten::pow_2616:
      type: Pow
      input:
        ret506.1: {}
        aten::pow_2616_other: {}
      output:
        ret507.1: {}
    aten::mean_985:
      type: ReduceMean
      input:
        ret507.1: {}
      output:
        ret508.1: {}
    aten::add_20:
      type: Add
      input:
        ret508.1: {}
        '485': {}
      output:
        ret509.1: {}
    aten::rsqrt_4688:
      type: Rsqrt
      input:
        ret509.1: {}
      output:
        ret510.1: {}
    aten::mul_4687:
      type: Mul
      input:
        ret506.1: {}
        ret510.1: {}
      output:
        ret511.1: {}
      attr:
        algorithm: mul
    aten::mul_425:
      type: Mul
      input:
        '587': {}
        ret511.1: {}
      output:
        ret512.1: {}
      attr:
        algorithm: mul
    aten::size_3395:
      type: Shape
      input:
        ret512.1: {}
      output:
        '2750': {}
      attr:
        start: 0
        end: 0
    aten::size_3109:
      type: Shape
      input:
        ret512.1: {}
      output:
        '2752': {}
      attr:
        start: 1
        end: 1
    aten::mul_426:
      type: Mul
      input:
        ret512.1: {}
        '588': {}
      output:
        ret515.1: {}
      attr:
        algorithm: mul
    aten::linear_3478:
      type: InnerProduct
      input:
        ret515.1: {}
        '589': {}
        aten::linear_3478_bias: {}
      output:
        ret518.1: {}
    aten::view_4690:
      type: View
      input:
        ret518.1: {}
        '2750': {}
        '2752': {}
      output:
        ret519.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2617:
      type: Reorder
      input:
        ret519.1: {}
      output:
        ret520.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_428:
      type: Mul
      input:
        ret512.1: {}
        '590': {}
      output:
        ret521.1: {}
      attr:
        algorithm: mul
    aten::linear_3479:
      type: InnerProduct
      input:
        ret521.1: {}
        '591': {}
        aten::linear_3479_bias: {}
      output:
        ret524.1: {}
    aten::view_4691:
      type: View
      input:
        ret524.1: {}
        '2750': {}
        '2752': {}
      output:
        ret525.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2618:
      type: Reorder
      input:
        ret525.1: {}
      output:
        ret526.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_430:
      type: Mul
      input:
        ret512.1: {}
        '592': {}
      output:
        ret527.1: {}
      attr:
        algorithm: mul
    aten::linear_3480:
      type: InnerProduct
      input:
        ret527.1: {}
        '593': {}
        aten::linear_3480_bias: {}
      output:
        ret530.1: {}
    aten::view_4692:
      type: View
      input:
        ret530.1: {}
        '2750': {}
        '2752': {}
      output:
        ret531.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2619:
      type: Reorder
      input:
        ret531.1: {}
      output:
        ret532.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2620:
      type: Shape
      input:
        ret526.1: {}
      output:
        '2798': {}
      attr:
        start: 2
        end: 2
    aten::size_2621:
      type: Shape
      input:
        x11.1: {}
      output:
        '2801': {}
      attr:
        start: 2
        end: 2
    aten::add_3353:
      type: Add
      input:
        '2798': {}
        '2801': {}
      output:
        seq_len5.1: {}
    aten::slice_102:
      type: Slice
      input:
        '493': {}
        seq_len5.1: {}
      output:
        '2808': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_142:
      type: Slice
      input:
        '494': {}
        seq_len5.1: {}
      output:
        '2810': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2622:
      type: Shape
      input:
        ret520.1: {}
      output:
        '2812': {}
      attr:
        start: 2
        end: 2
    aten::add_3110:
      type: Add
      input:
        '2812': {}
        '2801': {}
      output:
        '2817': {}
    aten::slice_2623:
      type: Slice
      input:
        '2808': {}
        '2801': {}
        '2817': {}
      output:
        '2819': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2099:
      type: Slice
      input:
        '2819': {}
      output:
        '2820': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2624:
      type: Slice
      input:
        '2810': {}
        '2801': {}
        '2817': {}
      output:
        '2821': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2100:
      type: Slice
      input:
        '2821': {}
      output:
        '2822': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4693:
      type: Mul
      input:
        ret520.1: {}
        '2820': {}
      output:
        ret536.1: {}
      attr:
        algorithm: mul
    aten::size_2101:
      type: Shape
      input:
        ret520.1: {}
      output:
        '2826': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_188:
      type: Div
      input:
        '2826': {}
        '495': {}
      output:
        ret538.1: {}
      attr:
        algorithm: div
    aten::slice_2102:
      type: Slice
      input:
        ret520.1: {}
        ret538.1: {}
      output:
        ret539.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2103:
      type: Slice
      input:
        ret520.1: {}
        ret538.1: {}
      output:
        ret540.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4700:
      type: Neg
      input:
        ret540.1: {}
        aten::neg_4700_mul_val: {}
      output:
        ret541.1: {}
      attr:
        algorithm: mul
    aten::cat_2465:
      type: Concat
      input:
        ret541.1: {}
        ret539.1: {}
      output:
        ret542.1: {}
      attr:
        axis: -1
    aten::mul_4697:
      type: Mul
      input:
        ret542.1: {}
        '2822': {}
      output:
        ret543.1: {}
      attr:
        algorithm: mul
    aten::add_3111:
      type: Add
      input:
        ret536.1: {}
        ret543.1: {}
      output:
        args23.1: {}
    aten::mul_4695:
      type: Mul
      input:
        ret526.1: {}
        '2820': {}
      output:
        ret544.1: {}
      attr:
        algorithm: mul
    aten::size_2104:
      type: Shape
      input:
        ret526.1: {}
      output:
        '2848': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_189:
      type: Div
      input:
        '2848': {}
        '495': {}
      output:
        ret546.1: {}
      attr:
        algorithm: div
    aten::slice_2105:
      type: Slice
      input:
        ret526.1: {}
        ret546.1: {}
      output:
        ret547.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2106:
      type: Slice
      input:
        ret526.1: {}
        ret546.1: {}
      output:
        ret548.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4702:
      type: Neg
      input:
        ret548.1: {}
        aten::neg_4702_mul_val: {}
      output:
        ret549.1: {}
      attr:
        algorithm: mul
    aten::cat_2466:
      type: Concat
      input:
        ret549.1: {}
        ret547.1: {}
      output:
        ret550.1: {}
      attr:
        axis: -1
    aten::mul_4698:
      type: Mul
      input:
        ret550.1: {}
        '2822': {}
      output:
        ret551.1: {}
      attr:
        algorithm: mul
    aten::add_3112:
      type: Add
      input:
        ret544.1: {}
        ret551.1: {}
      output:
        '2866': {}
    aten::cat_2625:
      type: Concat
      input:
        x11.1: {}
        '2866': {}
      output:
        ret552.1: {}
      attr:
        axis: 2
    aten::cat_2626:
      type: Concat
      input:
        x12.1: {}
        ret532.1: {}
      output:
        ret553.1: {}
      attr:
        axis: 2
    aten::transpose_2107:
      type: Reorder
      input:
        ret552.1: {}
      output:
        ret554.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4705:
      type: Matmul
      input:
        args23.1: {}
        ret554.1: {}
      output:
        ret557.1: {}
    aten::div_262:
      type: Div
      input:
        ret557.1: {}
        '496': {}
      output:
        ret558.1: {}
      attr:
        algorithm: div
    aten::add_3113:
      type: Add
      input:
        ret558.1: {}
        attention_mask0.1: {}
      output:
        attn_weights12.1: {}
    aten::max_302:
      type: Max
      input:
        attn_weights12.1: {}
        '497': {}
      output:
        input12.1: {}
    aten::softmax_2415:
      type: Softmax
      input:
        input12.1: {}
      output:
        '2888': {}
      attr:
        axis: -1
    aten::matmul_4708:
      type: Matmul
      input:
        '2888': {}
        ret553.1: {}
      output:
        ret561.1: {}
    aten::transpose_2627:
      type: Reorder
      input:
        ret561.1: {}
      output:
        ret562.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4710:
      type: Reshape
      input:
        ret562.1: {}
        '2750': {}
        '2752': {}
      output:
        ret563.1: {}
    aten::mul_432:
      type: Mul
      input:
        ret563.1: {}
        '594': {}
      output:
        ret564.1: {}
      attr:
        algorithm: mul
    aten::linear_3481:
      type: InnerProduct
      input:
        ret564.1: {}
        '595': {}
        aten::linear_3481_bias: {}
      output:
        ret567.1: {}
    aten::add_3114:
      type: Add
      input:
        ret506.1: {}
        ret567.1: {}
      output:
        ret568.1: {}
    aten::pow_2628:
      type: Pow
      input:
        ret568.1: {}
        aten::pow_2628_other: {}
      output:
        ret569.1: {}
    aten::mean_986:
      type: ReduceMean
      input:
        ret569.1: {}
      output:
        ret570.1: {}
    aten::add_21:
      type: Add
      input:
        ret570.1: {}
        '485': {}
      output:
        ret571.1: {}
    aten::rsqrt_4713:
      type: Rsqrt
      input:
        ret571.1: {}
      output:
        ret572.1: {}
    aten::mul_4712:
      type: Mul
      input:
        ret568.1: {}
        ret572.1: {}
      output:
        ret573.1: {}
      attr:
        algorithm: mul
    aten::mul_434:
      type: Mul
      input:
        '596': {}
        ret573.1: {}
      output:
        ret574.1: {}
      attr:
        algorithm: mul
    aten::mul_435:
      type: Mul
      input:
        ret574.1: {}
        '597': {}
      output:
        ret575.1: {}
      attr:
        algorithm: mul
    aten::linear_3482:
      type: InnerProduct
      input:
        ret575.1: {}
        '598': {}
        aten::linear_3482_bias: {}
      output:
        ret578.1: {}
    aten::silu_4715:
      type: Swish
      input:
        ret578.1: {}
      output:
        ret579.1: {}
    aten::mul_437:
      type: Mul
      input:
        ret574.1: {}
        '599': {}
      output:
        ret580.1: {}
      attr:
        algorithm: mul
    aten::linear_3483:
      type: InnerProduct
      input:
        ret580.1: {}
        '600': {}
        aten::linear_3483_bias: {}
      output:
        ret583.1: {}
    aten::mul_4716:
      type: Mul
      input:
        ret579.1: {}
        ret583.1: {}
      output:
        ret584.1: {}
      attr:
        algorithm: mul
    aten::mul_439:
      type: Mul
      input:
        ret584.1: {}
        '601': {}
      output:
        ret585.1: {}
      attr:
        algorithm: mul
    aten::linear_3484:
      type: InnerProduct
      input:
        ret585.1: {}
        '602': {}
        aten::linear_3484_bias: {}
      output:
        ret588.1: {}
    aten::add_3115:
      type: Add
      input:
        ret568.1: {}
        ret588.1: {}
      output:
        ret589.1: {}
    aten::pow_2629:
      type: Pow
      input:
        ret589.1: {}
        aten::pow_2629_other: {}
      output:
        ret590.1: {}
    aten::mean_987:
      type: ReduceMean
      input:
        ret590.1: {}
      output:
        ret591.1: {}
    aten::add_22:
      type: Add
      input:
        ret591.1: {}
        '485': {}
      output:
        ret592.1: {}
    aten::rsqrt_4720:
      type: Rsqrt
      input:
        ret592.1: {}
      output:
        ret593.1: {}
    aten::mul_4719:
      type: Mul
      input:
        ret589.1: {}
        ret593.1: {}
      output:
        ret594.1: {}
      attr:
        algorithm: mul
    aten::mul_441:
      type: Mul
      input:
        '603': {}
        ret594.1: {}
      output:
        ret595.1: {}
      attr:
        algorithm: mul
    aten::size_3396:
      type: Shape
      input:
        ret595.1: {}
      output:
        '2973': {}
      attr:
        start: 0
        end: 0
    aten::size_3116:
      type: Shape
      input:
        ret595.1: {}
      output:
        '2975': {}
      attr:
        start: 1
        end: 1
    aten::mul_442:
      type: Mul
      input:
        ret595.1: {}
        '604': {}
      output:
        ret598.1: {}
      attr:
        algorithm: mul
    aten::linear_3485:
      type: InnerProduct
      input:
        ret598.1: {}
        '605': {}
        aten::linear_3485_bias: {}
      output:
        ret601.1: {}
    aten::view_4722:
      type: View
      input:
        ret601.1: {}
        '2973': {}
        '2975': {}
      output:
        ret602.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2630:
      type: Reorder
      input:
        ret602.1: {}
      output:
        ret603.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_444:
      type: Mul
      input:
        ret595.1: {}
        '606': {}
      output:
        ret604.1: {}
      attr:
        algorithm: mul
    aten::linear_3486:
      type: InnerProduct
      input:
        ret604.1: {}
        '607': {}
        aten::linear_3486_bias: {}
      output:
        ret607.1: {}
    aten::view_4723:
      type: View
      input:
        ret607.1: {}
        '2973': {}
        '2975': {}
      output:
        ret608.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2631:
      type: Reorder
      input:
        ret608.1: {}
      output:
        ret609.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_446:
      type: Mul
      input:
        ret595.1: {}
        '608': {}
      output:
        ret610.1: {}
      attr:
        algorithm: mul
    aten::linear_3487:
      type: InnerProduct
      input:
        ret610.1: {}
        '609': {}
        aten::linear_3487_bias: {}
      output:
        ret613.1: {}
    aten::view_4724:
      type: View
      input:
        ret613.1: {}
        '2973': {}
        '2975': {}
      output:
        ret614.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2632:
      type: Reorder
      input:
        ret614.1: {}
      output:
        ret615.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2633:
      type: Shape
      input:
        ret609.1: {}
      output:
        '3021': {}
      attr:
        start: 2
        end: 2
    aten::size_2634:
      type: Shape
      input:
        x13.1: {}
      output:
        '3024': {}
      attr:
        start: 2
        end: 2
    aten::add_3354:
      type: Add
      input:
        '3021': {}
        '3024': {}
      output:
        seq_len6.1: {}
    aten::slice_103:
      type: Slice
      input:
        '493': {}
        seq_len6.1: {}
      output:
        '3031': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_143:
      type: Slice
      input:
        '494': {}
        seq_len6.1: {}
      output:
        '3033': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2635:
      type: Shape
      input:
        ret603.1: {}
      output:
        '3035': {}
      attr:
        start: 2
        end: 2
    aten::add_3117:
      type: Add
      input:
        '3035': {}
        '3024': {}
      output:
        '3040': {}
    aten::slice_2636:
      type: Slice
      input:
        '3031': {}
        '3024': {}
        '3040': {}
      output:
        '3042': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2108:
      type: Slice
      input:
        '3042': {}
      output:
        '3043': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2637:
      type: Slice
      input:
        '3033': {}
        '3024': {}
        '3040': {}
      output:
        '3044': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2109:
      type: Slice
      input:
        '3044': {}
      output:
        '3045': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4725:
      type: Mul
      input:
        ret603.1: {}
        '3043': {}
      output:
        ret619.1: {}
      attr:
        algorithm: mul
    aten::size_2110:
      type: Shape
      input:
        ret603.1: {}
      output:
        '3049': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_190:
      type: Div
      input:
        '3049': {}
        '495': {}
      output:
        ret621.1: {}
      attr:
        algorithm: div
    aten::slice_2111:
      type: Slice
      input:
        ret603.1: {}
        ret621.1: {}
      output:
        ret622.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2112:
      type: Slice
      input:
        ret603.1: {}
        ret621.1: {}
      output:
        ret623.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4732:
      type: Neg
      input:
        ret623.1: {}
        aten::neg_4732_mul_val: {}
      output:
        ret624.1: {}
      attr:
        algorithm: mul
    aten::cat_2467:
      type: Concat
      input:
        ret624.1: {}
        ret622.1: {}
      output:
        ret625.1: {}
      attr:
        axis: -1
    aten::mul_4729:
      type: Mul
      input:
        ret625.1: {}
        '3045': {}
      output:
        ret626.1: {}
      attr:
        algorithm: mul
    aten::add_3118:
      type: Add
      input:
        ret619.1: {}
        ret626.1: {}
      output:
        args27.1: {}
    aten::mul_4727:
      type: Mul
      input:
        ret609.1: {}
        '3043': {}
      output:
        ret627.1: {}
      attr:
        algorithm: mul
    aten::size_2113:
      type: Shape
      input:
        ret609.1: {}
      output:
        '3071': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_191:
      type: Div
      input:
        '3071': {}
        '495': {}
      output:
        ret629.1: {}
      attr:
        algorithm: div
    aten::slice_2114:
      type: Slice
      input:
        ret609.1: {}
        ret629.1: {}
      output:
        ret630.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2115:
      type: Slice
      input:
        ret609.1: {}
        ret629.1: {}
      output:
        ret631.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4734:
      type: Neg
      input:
        ret631.1: {}
        aten::neg_4734_mul_val: {}
      output:
        ret632.1: {}
      attr:
        algorithm: mul
    aten::cat_2468:
      type: Concat
      input:
        ret632.1: {}
        ret630.1: {}
      output:
        ret633.1: {}
      attr:
        axis: -1
    aten::mul_4730:
      type: Mul
      input:
        ret633.1: {}
        '3045': {}
      output:
        ret634.1: {}
      attr:
        algorithm: mul
    aten::add_3119:
      type: Add
      input:
        ret627.1: {}
        ret634.1: {}
      output:
        '3089': {}
    aten::cat_2638:
      type: Concat
      input:
        x13.1: {}
        '3089': {}
      output:
        ret635.1: {}
      attr:
        axis: 2
    aten::cat_2639:
      type: Concat
      input:
        x14.1: {}
        ret615.1: {}
      output:
        ret636.1: {}
      attr:
        axis: 2
    aten::transpose_2116:
      type: Reorder
      input:
        ret635.1: {}
      output:
        ret637.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4737:
      type: Matmul
      input:
        args27.1: {}
        ret637.1: {}
      output:
        ret640.1: {}
    aten::div_263:
      type: Div
      input:
        ret640.1: {}
        '496': {}
      output:
        ret641.1: {}
      attr:
        algorithm: div
    aten::add_3120:
      type: Add
      input:
        ret641.1: {}
        attention_mask0.1: {}
      output:
        attn_weights14.1: {}
    aten::max_303:
      type: Max
      input:
        attn_weights14.1: {}
        '497': {}
      output:
        input14.1: {}
    aten::softmax_2416:
      type: Softmax
      input:
        input14.1: {}
      output:
        '3111': {}
      attr:
        axis: -1
    aten::matmul_4740:
      type: Matmul
      input:
        '3111': {}
        ret636.1: {}
      output:
        ret644.1: {}
    aten::transpose_2640:
      type: Reorder
      input:
        ret644.1: {}
      output:
        ret645.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4742:
      type: Reshape
      input:
        ret645.1: {}
        '2973': {}
        '2975': {}
      output:
        ret646.1: {}
    aten::mul_448:
      type: Mul
      input:
        ret646.1: {}
        '610': {}
      output:
        ret647.1: {}
      attr:
        algorithm: mul
    aten::linear_3488:
      type: InnerProduct
      input:
        ret647.1: {}
        '611': {}
        aten::linear_3488_bias: {}
      output:
        ret650.1: {}
    aten::add_3121:
      type: Add
      input:
        ret589.1: {}
        ret650.1: {}
      output:
        ret651.1: {}
    aten::pow_2641:
      type: Pow
      input:
        ret651.1: {}
        aten::pow_2641_other: {}
      output:
        ret652.1: {}
    aten::mean_988:
      type: ReduceMean
      input:
        ret652.1: {}
      output:
        ret653.1: {}
    aten::add_23:
      type: Add
      input:
        ret653.1: {}
        '485': {}
      output:
        ret654.1: {}
    aten::rsqrt_4745:
      type: Rsqrt
      input:
        ret654.1: {}
      output:
        ret655.1: {}
    aten::mul_4744:
      type: Mul
      input:
        ret651.1: {}
        ret655.1: {}
      output:
        ret656.1: {}
      attr:
        algorithm: mul
    aten::mul_450:
      type: Mul
      input:
        '612': {}
        ret656.1: {}
      output:
        ret657.1: {}
      attr:
        algorithm: mul
    aten::mul_451:
      type: Mul
      input:
        ret657.1: {}
        '613': {}
      output:
        ret658.1: {}
      attr:
        algorithm: mul
    aten::linear_3489:
      type: InnerProduct
      input:
        ret658.1: {}
        '614': {}
        aten::linear_3489_bias: {}
      output:
        ret661.1: {}
    aten::silu_4747:
      type: Swish
      input:
        ret661.1: {}
      output:
        ret662.1: {}
    aten::mul_453:
      type: Mul
      input:
        ret657.1: {}
        '615': {}
      output:
        ret663.1: {}
      attr:
        algorithm: mul
    aten::linear_3490:
      type: InnerProduct
      input:
        ret663.1: {}
        '616': {}
        aten::linear_3490_bias: {}
      output:
        ret666.1: {}
    aten::mul_4748:
      type: Mul
      input:
        ret662.1: {}
        ret666.1: {}
      output:
        ret667.1: {}
      attr:
        algorithm: mul
    aten::mul_455:
      type: Mul
      input:
        ret667.1: {}
        '617': {}
      output:
        ret668.1: {}
      attr:
        algorithm: mul
    aten::linear_3491:
      type: InnerProduct
      input:
        ret668.1: {}
        '618': {}
        aten::linear_3491_bias: {}
      output:
        ret671.1: {}
    aten::add_3122:
      type: Add
      input:
        ret651.1: {}
        ret671.1: {}
      output:
        ret672.1: {}
    aten::pow_2642:
      type: Pow
      input:
        ret672.1: {}
        aten::pow_2642_other: {}
      output:
        ret673.1: {}
    aten::mean_989:
      type: ReduceMean
      input:
        ret673.1: {}
      output:
        ret674.1: {}
    aten::add_24:
      type: Add
      input:
        ret674.1: {}
        '485': {}
      output:
        ret675.1: {}
    aten::rsqrt_4752:
      type: Rsqrt
      input:
        ret675.1: {}
      output:
        ret676.1: {}
    aten::mul_4751:
      type: Mul
      input:
        ret672.1: {}
        ret676.1: {}
      output:
        ret677.1: {}
      attr:
        algorithm: mul
    aten::mul_457:
      type: Mul
      input:
        '619': {}
        ret677.1: {}
      output:
        ret678.1: {}
      attr:
        algorithm: mul
    aten::size_3397:
      type: Shape
      input:
        ret678.1: {}
      output:
        '3196': {}
      attr:
        start: 0
        end: 0
    aten::size_3123:
      type: Shape
      input:
        ret678.1: {}
      output:
        '3198': {}
      attr:
        start: 1
        end: 1
    aten::mul_458:
      type: Mul
      input:
        ret678.1: {}
        '620': {}
      output:
        ret681.1: {}
      attr:
        algorithm: mul
    aten::linear_3492:
      type: InnerProduct
      input:
        ret681.1: {}
        '621': {}
        aten::linear_3492_bias: {}
      output:
        ret684.1: {}
    aten::view_4754:
      type: View
      input:
        ret684.1: {}
        '3196': {}
        '3198': {}
      output:
        ret685.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2643:
      type: Reorder
      input:
        ret685.1: {}
      output:
        ret686.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_460:
      type: Mul
      input:
        ret678.1: {}
        '622': {}
      output:
        ret687.1: {}
      attr:
        algorithm: mul
    aten::linear_3493:
      type: InnerProduct
      input:
        ret687.1: {}
        '623': {}
        aten::linear_3493_bias: {}
      output:
        ret690.1: {}
    aten::view_4755:
      type: View
      input:
        ret690.1: {}
        '3196': {}
        '3198': {}
      output:
        ret691.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2644:
      type: Reorder
      input:
        ret691.1: {}
      output:
        ret692.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_462:
      type: Mul
      input:
        ret678.1: {}
        '624': {}
      output:
        ret693.1: {}
      attr:
        algorithm: mul
    aten::linear_3494:
      type: InnerProduct
      input:
        ret693.1: {}
        '625': {}
        aten::linear_3494_bias: {}
      output:
        ret696.1: {}
    aten::view_4756:
      type: View
      input:
        ret696.1: {}
        '3196': {}
        '3198': {}
      output:
        ret697.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2645:
      type: Reorder
      input:
        ret697.1: {}
      output:
        ret698.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2646:
      type: Shape
      input:
        ret692.1: {}
      output:
        '3244': {}
      attr:
        start: 2
        end: 2
    aten::size_2647:
      type: Shape
      input:
        x15.1: {}
      output:
        '3247': {}
      attr:
        start: 2
        end: 2
    aten::add_3355:
      type: Add
      input:
        '3244': {}
        '3247': {}
      output:
        seq_len7.1: {}
    aten::slice_104:
      type: Slice
      input:
        '493': {}
        seq_len7.1: {}
      output:
        '3254': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_144:
      type: Slice
      input:
        '494': {}
        seq_len7.1: {}
      output:
        '3256': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2648:
      type: Shape
      input:
        ret686.1: {}
      output:
        '3258': {}
      attr:
        start: 2
        end: 2
    aten::add_3124:
      type: Add
      input:
        '3258': {}
        '3247': {}
      output:
        '3263': {}
    aten::slice_2649:
      type: Slice
      input:
        '3254': {}
        '3247': {}
        '3263': {}
      output:
        '3265': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2117:
      type: Slice
      input:
        '3265': {}
      output:
        '3266': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2650:
      type: Slice
      input:
        '3256': {}
        '3247': {}
        '3263': {}
      output:
        '3267': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2118:
      type: Slice
      input:
        '3267': {}
      output:
        '3268': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4757:
      type: Mul
      input:
        ret686.1: {}
        '3266': {}
      output:
        ret702.1: {}
      attr:
        algorithm: mul
    aten::size_2119:
      type: Shape
      input:
        ret686.1: {}
      output:
        '3272': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_192:
      type: Div
      input:
        '3272': {}
        '495': {}
      output:
        ret704.1: {}
      attr:
        algorithm: div
    aten::slice_2120:
      type: Slice
      input:
        ret686.1: {}
        ret704.1: {}
      output:
        ret705.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2121:
      type: Slice
      input:
        ret686.1: {}
        ret704.1: {}
      output:
        ret706.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4764:
      type: Neg
      input:
        ret706.1: {}
        aten::neg_4764_mul_val: {}
      output:
        ret707.1: {}
      attr:
        algorithm: mul
    aten::cat_2469:
      type: Concat
      input:
        ret707.1: {}
        ret705.1: {}
      output:
        ret708.1: {}
      attr:
        axis: -1
    aten::mul_4761:
      type: Mul
      input:
        ret708.1: {}
        '3268': {}
      output:
        ret709.1: {}
      attr:
        algorithm: mul
    aten::add_3125:
      type: Add
      input:
        ret702.1: {}
        ret709.1: {}
      output:
        args31.1: {}
    aten::mul_4759:
      type: Mul
      input:
        ret692.1: {}
        '3266': {}
      output:
        ret710.1: {}
      attr:
        algorithm: mul
    aten::size_2122:
      type: Shape
      input:
        ret692.1: {}
      output:
        '3294': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_193:
      type: Div
      input:
        '3294': {}
        '495': {}
      output:
        ret712.1: {}
      attr:
        algorithm: div
    aten::slice_2123:
      type: Slice
      input:
        ret692.1: {}
        ret712.1: {}
      output:
        ret713.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2124:
      type: Slice
      input:
        ret692.1: {}
        ret712.1: {}
      output:
        ret714.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4766:
      type: Neg
      input:
        ret714.1: {}
        aten::neg_4766_mul_val: {}
      output:
        ret715.1: {}
      attr:
        algorithm: mul
    aten::cat_2470:
      type: Concat
      input:
        ret715.1: {}
        ret713.1: {}
      output:
        ret716.1: {}
      attr:
        axis: -1
    aten::mul_4762:
      type: Mul
      input:
        ret716.1: {}
        '3268': {}
      output:
        ret717.1: {}
      attr:
        algorithm: mul
    aten::add_3126:
      type: Add
      input:
        ret710.1: {}
        ret717.1: {}
      output:
        '3312': {}
    aten::cat_2651:
      type: Concat
      input:
        x15.1: {}
        '3312': {}
      output:
        ret718.1: {}
      attr:
        axis: 2
    aten::cat_2652:
      type: Concat
      input:
        x16.1: {}
        ret698.1: {}
      output:
        ret719.1: {}
      attr:
        axis: 2
    aten::transpose_2125:
      type: Reorder
      input:
        ret718.1: {}
      output:
        ret720.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4769:
      type: Matmul
      input:
        args31.1: {}
        ret720.1: {}
      output:
        ret723.1: {}
    aten::div_264:
      type: Div
      input:
        ret723.1: {}
        '496': {}
      output:
        ret724.1: {}
      attr:
        algorithm: div
    aten::add_3127:
      type: Add
      input:
        ret724.1: {}
        attention_mask0.1: {}
      output:
        attn_weights16.1: {}
    aten::max_304:
      type: Max
      input:
        attn_weights16.1: {}
        '497': {}
      output:
        input16.1: {}
    aten::softmax_2417:
      type: Softmax
      input:
        input16.1: {}
      output:
        '3334': {}
      attr:
        axis: -1
    aten::matmul_4772:
      type: Matmul
      input:
        '3334': {}
        ret719.1: {}
      output:
        ret727.1: {}
    aten::transpose_2653:
      type: Reorder
      input:
        ret727.1: {}
      output:
        ret728.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4774:
      type: Reshape
      input:
        ret728.1: {}
        '3196': {}
        '3198': {}
      output:
        ret729.1: {}
    aten::mul_464:
      type: Mul
      input:
        ret729.1: {}
        '626': {}
      output:
        ret730.1: {}
      attr:
        algorithm: mul
    aten::linear_3495:
      type: InnerProduct
      input:
        ret730.1: {}
        '627': {}
        aten::linear_3495_bias: {}
      output:
        ret733.1: {}
    aten::add_3128:
      type: Add
      input:
        ret672.1: {}
        ret733.1: {}
      output:
        ret734.1: {}
    aten::pow_2654:
      type: Pow
      input:
        ret734.1: {}
        aten::pow_2654_other: {}
      output:
        ret735.1: {}
    aten::mean_990:
      type: ReduceMean
      input:
        ret735.1: {}
      output:
        ret736.1: {}
    aten::add_25:
      type: Add
      input:
        ret736.1: {}
        '485': {}
      output:
        ret737.1: {}
    aten::rsqrt_4777:
      type: Rsqrt
      input:
        ret737.1: {}
      output:
        ret738.1: {}
    aten::mul_4776:
      type: Mul
      input:
        ret734.1: {}
        ret738.1: {}
      output:
        ret739.1: {}
      attr:
        algorithm: mul
    aten::mul_466:
      type: Mul
      input:
        '628': {}
        ret739.1: {}
      output:
        ret740.1: {}
      attr:
        algorithm: mul
    aten::mul_467:
      type: Mul
      input:
        ret740.1: {}
        '629': {}
      output:
        ret741.1: {}
      attr:
        algorithm: mul
    aten::linear_3496:
      type: InnerProduct
      input:
        ret741.1: {}
        '630': {}
        aten::linear_3496_bias: {}
      output:
        ret744.1: {}
    aten::silu_4779:
      type: Swish
      input:
        ret744.1: {}
      output:
        ret745.1: {}
    aten::mul_469:
      type: Mul
      input:
        ret740.1: {}
        '631': {}
      output:
        ret746.1: {}
      attr:
        algorithm: mul
    aten::linear_3497:
      type: InnerProduct
      input:
        ret746.1: {}
        '632': {}
        aten::linear_3497_bias: {}
      output:
        ret749.1: {}
    aten::mul_4780:
      type: Mul
      input:
        ret745.1: {}
        ret749.1: {}
      output:
        ret750.1: {}
      attr:
        algorithm: mul
    aten::mul_471:
      type: Mul
      input:
        ret750.1: {}
        '633': {}
      output:
        ret751.1: {}
      attr:
        algorithm: mul
    aten::linear_3498:
      type: InnerProduct
      input:
        ret751.1: {}
        '634': {}
        aten::linear_3498_bias: {}
      output:
        ret754.1: {}
    aten::add_3129:
      type: Add
      input:
        ret734.1: {}
        ret754.1: {}
      output:
        ret755.1: {}
    aten::pow_2655:
      type: Pow
      input:
        ret755.1: {}
        aten::pow_2655_other: {}
      output:
        ret756.1: {}
    aten::mean_991:
      type: ReduceMean
      input:
        ret756.1: {}
      output:
        ret757.1: {}
    aten::add_26:
      type: Add
      input:
        ret757.1: {}
        '485': {}
      output:
        ret758.1: {}
    aten::rsqrt_4784:
      type: Rsqrt
      input:
        ret758.1: {}
      output:
        ret759.1: {}
    aten::mul_4783:
      type: Mul
      input:
        ret755.1: {}
        ret759.1: {}
      output:
        ret760.1: {}
      attr:
        algorithm: mul
    aten::mul_473:
      type: Mul
      input:
        '635': {}
        ret760.1: {}
      output:
        ret761.1: {}
      attr:
        algorithm: mul
    aten::size_3398:
      type: Shape
      input:
        ret761.1: {}
      output:
        '3419': {}
      attr:
        start: 0
        end: 0
    aten::size_3130:
      type: Shape
      input:
        ret761.1: {}
      output:
        '3421': {}
      attr:
        start: 1
        end: 1
    aten::mul_474:
      type: Mul
      input:
        ret761.1: {}
        '636': {}
      output:
        ret764.1: {}
      attr:
        algorithm: mul
    aten::linear_3499:
      type: InnerProduct
      input:
        ret764.1: {}
        '637': {}
        aten::linear_3499_bias: {}
      output:
        ret767.1: {}
    aten::view_4786:
      type: View
      input:
        ret767.1: {}
        '3419': {}
        '3421': {}
      output:
        ret768.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2656:
      type: Reorder
      input:
        ret768.1: {}
      output:
        ret769.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_476:
      type: Mul
      input:
        ret761.1: {}
        '638': {}
      output:
        ret770.1: {}
      attr:
        algorithm: mul
    aten::linear_3500:
      type: InnerProduct
      input:
        ret770.1: {}
        '639': {}
        aten::linear_3500_bias: {}
      output:
        ret773.1: {}
    aten::view_4787:
      type: View
      input:
        ret773.1: {}
        '3419': {}
        '3421': {}
      output:
        ret774.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2657:
      type: Reorder
      input:
        ret774.1: {}
      output:
        ret775.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_478:
      type: Mul
      input:
        ret761.1: {}
        '640': {}
      output:
        ret776.1: {}
      attr:
        algorithm: mul
    aten::linear_3501:
      type: InnerProduct
      input:
        ret776.1: {}
        '641': {}
        aten::linear_3501_bias: {}
      output:
        ret779.1: {}
    aten::view_4788:
      type: View
      input:
        ret779.1: {}
        '3419': {}
        '3421': {}
      output:
        ret780.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2658:
      type: Reorder
      input:
        ret780.1: {}
      output:
        ret781.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2659:
      type: Shape
      input:
        ret775.1: {}
      output:
        '3467': {}
      attr:
        start: 2
        end: 2
    aten::size_2660:
      type: Shape
      input:
        x17.1: {}
      output:
        '3470': {}
      attr:
        start: 2
        end: 2
    aten::add_3356:
      type: Add
      input:
        '3467': {}
        '3470': {}
      output:
        seq_len8.1: {}
    aten::slice_105:
      type: Slice
      input:
        '493': {}
        seq_len8.1: {}
      output:
        '3477': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_145:
      type: Slice
      input:
        '494': {}
        seq_len8.1: {}
      output:
        '3479': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2661:
      type: Shape
      input:
        ret769.1: {}
      output:
        '3481': {}
      attr:
        start: 2
        end: 2
    aten::add_3131:
      type: Add
      input:
        '3481': {}
        '3470': {}
      output:
        '3486': {}
    aten::slice_2662:
      type: Slice
      input:
        '3477': {}
        '3470': {}
        '3486': {}
      output:
        '3488': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2126:
      type: Slice
      input:
        '3488': {}
      output:
        '3489': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2663:
      type: Slice
      input:
        '3479': {}
        '3470': {}
        '3486': {}
      output:
        '3490': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2127:
      type: Slice
      input:
        '3490': {}
      output:
        '3491': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4789:
      type: Mul
      input:
        ret769.1: {}
        '3489': {}
      output:
        ret785.1: {}
      attr:
        algorithm: mul
    aten::size_2128:
      type: Shape
      input:
        ret769.1: {}
      output:
        '3495': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_194:
      type: Div
      input:
        '3495': {}
        '495': {}
      output:
        ret787.1: {}
      attr:
        algorithm: div
    aten::slice_2129:
      type: Slice
      input:
        ret769.1: {}
        ret787.1: {}
      output:
        ret788.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2130:
      type: Slice
      input:
        ret769.1: {}
        ret787.1: {}
      output:
        ret789.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4796:
      type: Neg
      input:
        ret789.1: {}
        aten::neg_4796_mul_val: {}
      output:
        ret790.1: {}
      attr:
        algorithm: mul
    aten::cat_2471:
      type: Concat
      input:
        ret790.1: {}
        ret788.1: {}
      output:
        ret791.1: {}
      attr:
        axis: -1
    aten::mul_4793:
      type: Mul
      input:
        ret791.1: {}
        '3491': {}
      output:
        ret792.1: {}
      attr:
        algorithm: mul
    aten::add_3132:
      type: Add
      input:
        ret785.1: {}
        ret792.1: {}
      output:
        args35.1: {}
    aten::mul_4791:
      type: Mul
      input:
        ret775.1: {}
        '3489': {}
      output:
        ret793.1: {}
      attr:
        algorithm: mul
    aten::size_2131:
      type: Shape
      input:
        ret775.1: {}
      output:
        '3517': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_195:
      type: Div
      input:
        '3517': {}
        '495': {}
      output:
        ret795.1: {}
      attr:
        algorithm: div
    aten::slice_2132:
      type: Slice
      input:
        ret775.1: {}
        ret795.1: {}
      output:
        ret796.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2133:
      type: Slice
      input:
        ret775.1: {}
        ret795.1: {}
      output:
        ret797.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4798:
      type: Neg
      input:
        ret797.1: {}
        aten::neg_4798_mul_val: {}
      output:
        ret798.1: {}
      attr:
        algorithm: mul
    aten::cat_2472:
      type: Concat
      input:
        ret798.1: {}
        ret796.1: {}
      output:
        ret799.1: {}
      attr:
        axis: -1
    aten::mul_4794:
      type: Mul
      input:
        ret799.1: {}
        '3491': {}
      output:
        ret800.1: {}
      attr:
        algorithm: mul
    aten::add_3133:
      type: Add
      input:
        ret793.1: {}
        ret800.1: {}
      output:
        '3535': {}
    aten::cat_2664:
      type: Concat
      input:
        x17.1: {}
        '3535': {}
      output:
        ret801.1: {}
      attr:
        axis: 2
    aten::cat_2665:
      type: Concat
      input:
        x18.1: {}
        ret781.1: {}
      output:
        ret802.1: {}
      attr:
        axis: 2
    aten::transpose_2134:
      type: Reorder
      input:
        ret801.1: {}
      output:
        ret803.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4801:
      type: Matmul
      input:
        args35.1: {}
        ret803.1: {}
      output:
        ret806.1: {}
    aten::div_265:
      type: Div
      input:
        ret806.1: {}
        '496': {}
      output:
        ret807.1: {}
      attr:
        algorithm: div
    aten::add_3134:
      type: Add
      input:
        ret807.1: {}
        attention_mask0.1: {}
      output:
        attn_weights18.1: {}
    aten::max_305:
      type: Max
      input:
        attn_weights18.1: {}
        '497': {}
      output:
        input18.1: {}
    aten::softmax_2418:
      type: Softmax
      input:
        input18.1: {}
      output:
        '3557': {}
      attr:
        axis: -1
    aten::matmul_4804:
      type: Matmul
      input:
        '3557': {}
        ret802.1: {}
      output:
        ret810.1: {}
    aten::transpose_2666:
      type: Reorder
      input:
        ret810.1: {}
      output:
        ret811.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4806:
      type: Reshape
      input:
        ret811.1: {}
        '3419': {}
        '3421': {}
      output:
        ret812.1: {}
    aten::mul_480:
      type: Mul
      input:
        ret812.1: {}
        '642': {}
      output:
        ret813.1: {}
      attr:
        algorithm: mul
    aten::linear_3502:
      type: InnerProduct
      input:
        ret813.1: {}
        '643': {}
        aten::linear_3502_bias: {}
      output:
        ret816.1: {}
    aten::add_3135:
      type: Add
      input:
        ret755.1: {}
        ret816.1: {}
      output:
        ret817.1: {}
    aten::pow_2667:
      type: Pow
      input:
        ret817.1: {}
        aten::pow_2667_other: {}
      output:
        ret818.1: {}
    aten::mean_992:
      type: ReduceMean
      input:
        ret818.1: {}
      output:
        ret819.1: {}
    aten::add_27:
      type: Add
      input:
        ret819.1: {}
        '485': {}
      output:
        ret820.1: {}
    aten::rsqrt_4809:
      type: Rsqrt
      input:
        ret820.1: {}
      output:
        ret821.1: {}
    aten::mul_4808:
      type: Mul
      input:
        ret817.1: {}
        ret821.1: {}
      output:
        ret822.1: {}
      attr:
        algorithm: mul
    aten::mul_482:
      type: Mul
      input:
        '644': {}
        ret822.1: {}
      output:
        ret823.1: {}
      attr:
        algorithm: mul
    aten::mul_483:
      type: Mul
      input:
        ret823.1: {}
        '645': {}
      output:
        ret824.1: {}
      attr:
        algorithm: mul
    aten::linear_3503:
      type: InnerProduct
      input:
        ret824.1: {}
        '646': {}
        aten::linear_3503_bias: {}
      output:
        ret827.1: {}
    aten::silu_4811:
      type: Swish
      input:
        ret827.1: {}
      output:
        ret828.1: {}
    aten::mul_485:
      type: Mul
      input:
        ret823.1: {}
        '647': {}
      output:
        ret829.1: {}
      attr:
        algorithm: mul
    aten::linear_3504:
      type: InnerProduct
      input:
        ret829.1: {}
        '648': {}
        aten::linear_3504_bias: {}
      output:
        ret832.1: {}
    aten::mul_4812:
      type: Mul
      input:
        ret828.1: {}
        ret832.1: {}
      output:
        ret833.1: {}
      attr:
        algorithm: mul
    aten::mul_487:
      type: Mul
      input:
        ret833.1: {}
        '649': {}
      output:
        ret834.1: {}
      attr:
        algorithm: mul
    aten::linear_3505:
      type: InnerProduct
      input:
        ret834.1: {}
        '650': {}
        aten::linear_3505_bias: {}
      output:
        ret837.1: {}
    aten::add_3136:
      type: Add
      input:
        ret817.1: {}
        ret837.1: {}
      output:
        ret838.1: {}
    aten::pow_2668:
      type: Pow
      input:
        ret838.1: {}
        aten::pow_2668_other: {}
      output:
        ret839.1: {}
    aten::mean_993:
      type: ReduceMean
      input:
        ret839.1: {}
      output:
        ret840.1: {}
    aten::add_28:
      type: Add
      input:
        ret840.1: {}
        '485': {}
      output:
        ret841.1: {}
    aten::rsqrt_4816:
      type: Rsqrt
      input:
        ret841.1: {}
      output:
        ret842.1: {}
    aten::mul_4815:
      type: Mul
      input:
        ret838.1: {}
        ret842.1: {}
      output:
        ret843.1: {}
      attr:
        algorithm: mul
    aten::mul_489:
      type: Mul
      input:
        '651': {}
        ret843.1: {}
      output:
        ret844.1: {}
      attr:
        algorithm: mul
    aten::size_3399:
      type: Shape
      input:
        ret844.1: {}
      output:
        '3642': {}
      attr:
        start: 0
        end: 0
    aten::size_3137:
      type: Shape
      input:
        ret844.1: {}
      output:
        '3644': {}
      attr:
        start: 1
        end: 1
    aten::mul_490:
      type: Mul
      input:
        ret844.1: {}
        '652': {}
      output:
        ret847.1: {}
      attr:
        algorithm: mul
    aten::linear_3506:
      type: InnerProduct
      input:
        ret847.1: {}
        '653': {}
        aten::linear_3506_bias: {}
      output:
        ret850.1: {}
    aten::view_4818:
      type: View
      input:
        ret850.1: {}
        '3642': {}
        '3644': {}
      output:
        ret851.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2669:
      type: Reorder
      input:
        ret851.1: {}
      output:
        ret852.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_492:
      type: Mul
      input:
        ret844.1: {}
        '654': {}
      output:
        ret853.1: {}
      attr:
        algorithm: mul
    aten::linear_3507:
      type: InnerProduct
      input:
        ret853.1: {}
        '655': {}
        aten::linear_3507_bias: {}
      output:
        ret856.1: {}
    aten::view_4819:
      type: View
      input:
        ret856.1: {}
        '3642': {}
        '3644': {}
      output:
        ret857.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2670:
      type: Reorder
      input:
        ret857.1: {}
      output:
        ret858.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_494:
      type: Mul
      input:
        ret844.1: {}
        '656': {}
      output:
        ret859.1: {}
      attr:
        algorithm: mul
    aten::linear_3508:
      type: InnerProduct
      input:
        ret859.1: {}
        '657': {}
        aten::linear_3508_bias: {}
      output:
        ret862.1: {}
    aten::view_4820:
      type: View
      input:
        ret862.1: {}
        '3642': {}
        '3644': {}
      output:
        ret863.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2671:
      type: Reorder
      input:
        ret863.1: {}
      output:
        ret864.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2672:
      type: Shape
      input:
        ret858.1: {}
      output:
        '3690': {}
      attr:
        start: 2
        end: 2
    aten::size_2673:
      type: Shape
      input:
        x19.1: {}
      output:
        '3693': {}
      attr:
        start: 2
        end: 2
    aten::add_3357:
      type: Add
      input:
        '3690': {}
        '3693': {}
      output:
        seq_len9.1: {}
    aten::slice_106:
      type: Slice
      input:
        '493': {}
        seq_len9.1: {}
      output:
        '3700': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_146:
      type: Slice
      input:
        '494': {}
        seq_len9.1: {}
      output:
        '3702': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2674:
      type: Shape
      input:
        ret852.1: {}
      output:
        '3704': {}
      attr:
        start: 2
        end: 2
    aten::add_3138:
      type: Add
      input:
        '3704': {}
        '3693': {}
      output:
        '3709': {}
    aten::slice_2675:
      type: Slice
      input:
        '3700': {}
        '3693': {}
        '3709': {}
      output:
        '3711': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2135:
      type: Slice
      input:
        '3711': {}
      output:
        '3712': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2676:
      type: Slice
      input:
        '3702': {}
        '3693': {}
        '3709': {}
      output:
        '3713': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2136:
      type: Slice
      input:
        '3713': {}
      output:
        '3714': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4821:
      type: Mul
      input:
        ret852.1: {}
        '3712': {}
      output:
        ret868.1: {}
      attr:
        algorithm: mul
    aten::size_2137:
      type: Shape
      input:
        ret852.1: {}
      output:
        '3718': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_196:
      type: Div
      input:
        '3718': {}
        '495': {}
      output:
        ret870.1: {}
      attr:
        algorithm: div
    aten::slice_2138:
      type: Slice
      input:
        ret852.1: {}
        ret870.1: {}
      output:
        ret871.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2139:
      type: Slice
      input:
        ret852.1: {}
        ret870.1: {}
      output:
        ret872.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4828:
      type: Neg
      input:
        ret872.1: {}
        aten::neg_4828_mul_val: {}
      output:
        ret873.1: {}
      attr:
        algorithm: mul
    aten::cat_2473:
      type: Concat
      input:
        ret873.1: {}
        ret871.1: {}
      output:
        ret874.1: {}
      attr:
        axis: -1
    aten::mul_4825:
      type: Mul
      input:
        ret874.1: {}
        '3714': {}
      output:
        ret875.1: {}
      attr:
        algorithm: mul
    aten::add_3139:
      type: Add
      input:
        ret868.1: {}
        ret875.1: {}
      output:
        args39.1: {}
    aten::mul_4823:
      type: Mul
      input:
        ret858.1: {}
        '3712': {}
      output:
        ret876.1: {}
      attr:
        algorithm: mul
    aten::size_2140:
      type: Shape
      input:
        ret858.1: {}
      output:
        '3740': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_197:
      type: Div
      input:
        '3740': {}
        '495': {}
      output:
        ret878.1: {}
      attr:
        algorithm: div
    aten::slice_2141:
      type: Slice
      input:
        ret858.1: {}
        ret878.1: {}
      output:
        ret879.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2142:
      type: Slice
      input:
        ret858.1: {}
        ret878.1: {}
      output:
        ret880.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4830:
      type: Neg
      input:
        ret880.1: {}
        aten::neg_4830_mul_val: {}
      output:
        ret881.1: {}
      attr:
        algorithm: mul
    aten::cat_2474:
      type: Concat
      input:
        ret881.1: {}
        ret879.1: {}
      output:
        ret882.1: {}
      attr:
        axis: -1
    aten::mul_4826:
      type: Mul
      input:
        ret882.1: {}
        '3714': {}
      output:
        ret883.1: {}
      attr:
        algorithm: mul
    aten::add_3140:
      type: Add
      input:
        ret876.1: {}
        ret883.1: {}
      output:
        '3758': {}
    aten::cat_2677:
      type: Concat
      input:
        x19.1: {}
        '3758': {}
      output:
        ret884.1: {}
      attr:
        axis: 2
    aten::cat_2678:
      type: Concat
      input:
        x20.1: {}
        ret864.1: {}
      output:
        ret885.1: {}
      attr:
        axis: 2
    aten::transpose_2143:
      type: Reorder
      input:
        ret884.1: {}
      output:
        ret886.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4833:
      type: Matmul
      input:
        args39.1: {}
        ret886.1: {}
      output:
        ret889.1: {}
    aten::div_266:
      type: Div
      input:
        ret889.1: {}
        '496': {}
      output:
        ret890.1: {}
      attr:
        algorithm: div
    aten::add_3141:
      type: Add
      input:
        ret890.1: {}
        attention_mask0.1: {}
      output:
        attn_weights20.1: {}
    aten::max_306:
      type: Max
      input:
        attn_weights20.1: {}
        '497': {}
      output:
        input20.1: {}
    aten::softmax_2419:
      type: Softmax
      input:
        input20.1: {}
      output:
        '3780': {}
      attr:
        axis: -1
    aten::matmul_4836:
      type: Matmul
      input:
        '3780': {}
        ret885.1: {}
      output:
        ret893.1: {}
    aten::transpose_2679:
      type: Reorder
      input:
        ret893.1: {}
      output:
        ret894.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4838:
      type: Reshape
      input:
        ret894.1: {}
        '3642': {}
        '3644': {}
      output:
        ret895.1: {}
    aten::mul_496:
      type: Mul
      input:
        ret895.1: {}
        '658': {}
      output:
        ret896.1: {}
      attr:
        algorithm: mul
    aten::linear_3509:
      type: InnerProduct
      input:
        ret896.1: {}
        '659': {}
        aten::linear_3509_bias: {}
      output:
        ret899.1: {}
    aten::add_3142:
      type: Add
      input:
        ret838.1: {}
        ret899.1: {}
      output:
        ret900.1: {}
    aten::pow_2680:
      type: Pow
      input:
        ret900.1: {}
        aten::pow_2680_other: {}
      output:
        ret901.1: {}
    aten::mean_994:
      type: ReduceMean
      input:
        ret901.1: {}
      output:
        ret902.1: {}
    aten::add_29:
      type: Add
      input:
        ret902.1: {}
        '485': {}
      output:
        ret903.1: {}
    aten::rsqrt_4841:
      type: Rsqrt
      input:
        ret903.1: {}
      output:
        ret904.1: {}
    aten::mul_4840:
      type: Mul
      input:
        ret900.1: {}
        ret904.1: {}
      output:
        ret905.1: {}
      attr:
        algorithm: mul
    aten::mul_498:
      type: Mul
      input:
        '660': {}
        ret905.1: {}
      output:
        ret906.1: {}
      attr:
        algorithm: mul
    aten::mul_499:
      type: Mul
      input:
        ret906.1: {}
        '661': {}
      output:
        ret907.1: {}
      attr:
        algorithm: mul
    aten::linear_3510:
      type: InnerProduct
      input:
        ret907.1: {}
        '662': {}
        aten::linear_3510_bias: {}
      output:
        ret910.1: {}
    aten::silu_4843:
      type: Swish
      input:
        ret910.1: {}
      output:
        ret911.1: {}
    aten::mul_501:
      type: Mul
      input:
        ret906.1: {}
        '663': {}
      output:
        ret912.1: {}
      attr:
        algorithm: mul
    aten::linear_3511:
      type: InnerProduct
      input:
        ret912.1: {}
        '664': {}
        aten::linear_3511_bias: {}
      output:
        ret915.1: {}
    aten::mul_4844:
      type: Mul
      input:
        ret911.1: {}
        ret915.1: {}
      output:
        ret916.1: {}
      attr:
        algorithm: mul
    aten::mul_503:
      type: Mul
      input:
        ret916.1: {}
        '665': {}
      output:
        ret917.1: {}
      attr:
        algorithm: mul
    aten::linear_3512:
      type: InnerProduct
      input:
        ret917.1: {}
        '666': {}
        aten::linear_3512_bias: {}
      output:
        ret920.1: {}
    aten::add_3143:
      type: Add
      input:
        ret900.1: {}
        ret920.1: {}
      output:
        ret921.1: {}
    aten::pow_2681:
      type: Pow
      input:
        ret921.1: {}
        aten::pow_2681_other: {}
      output:
        ret922.1: {}
    aten::mean_995:
      type: ReduceMean
      input:
        ret922.1: {}
      output:
        ret923.1: {}
    aten::add_30:
      type: Add
      input:
        ret923.1: {}
        '485': {}
      output:
        ret924.1: {}
    aten::rsqrt_4848:
      type: Rsqrt
      input:
        ret924.1: {}
      output:
        ret925.1: {}
    aten::mul_4847:
      type: Mul
      input:
        ret921.1: {}
        ret925.1: {}
      output:
        ret926.1: {}
      attr:
        algorithm: mul
    aten::mul_505:
      type: Mul
      input:
        '667': {}
        ret926.1: {}
      output:
        ret927.1: {}
      attr:
        algorithm: mul
    aten::size_3400:
      type: Shape
      input:
        ret927.1: {}
      output:
        '3865': {}
      attr:
        start: 0
        end: 0
    aten::size_3144:
      type: Shape
      input:
        ret927.1: {}
      output:
        '3867': {}
      attr:
        start: 1
        end: 1
    aten::mul_506:
      type: Mul
      input:
        ret927.1: {}
        '668': {}
      output:
        ret930.1: {}
      attr:
        algorithm: mul
    aten::linear_3513:
      type: InnerProduct
      input:
        ret930.1: {}
        '669': {}
        aten::linear_3513_bias: {}
      output:
        ret933.1: {}
    aten::view_4850:
      type: View
      input:
        ret933.1: {}
        '3865': {}
        '3867': {}
      output:
        ret934.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2682:
      type: Reorder
      input:
        ret934.1: {}
      output:
        ret935.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_508:
      type: Mul
      input:
        ret927.1: {}
        '670': {}
      output:
        ret936.1: {}
      attr:
        algorithm: mul
    aten::linear_3514:
      type: InnerProduct
      input:
        ret936.1: {}
        '671': {}
        aten::linear_3514_bias: {}
      output:
        ret939.1: {}
    aten::view_4851:
      type: View
      input:
        ret939.1: {}
        '3865': {}
        '3867': {}
      output:
        ret940.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2683:
      type: Reorder
      input:
        ret940.1: {}
      output:
        ret941.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_510:
      type: Mul
      input:
        ret927.1: {}
        '672': {}
      output:
        ret942.1: {}
      attr:
        algorithm: mul
    aten::linear_3515:
      type: InnerProduct
      input:
        ret942.1: {}
        '673': {}
        aten::linear_3515_bias: {}
      output:
        ret945.1: {}
    aten::view_4852:
      type: View
      input:
        ret945.1: {}
        '3865': {}
        '3867': {}
      output:
        ret946.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2684:
      type: Reorder
      input:
        ret946.1: {}
      output:
        ret947.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2685:
      type: Shape
      input:
        ret941.1: {}
      output:
        '3913': {}
      attr:
        start: 2
        end: 2
    aten::size_2686:
      type: Shape
      input:
        x21.1: {}
      output:
        '3916': {}
      attr:
        start: 2
        end: 2
    aten::add_3358:
      type: Add
      input:
        '3913': {}
        '3916': {}
      output:
        seq_len10.1: {}
    aten::slice_107:
      type: Slice
      input:
        '493': {}
        seq_len10.1: {}
      output:
        '3923': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_147:
      type: Slice
      input:
        '494': {}
        seq_len10.1: {}
      output:
        '3925': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2687:
      type: Shape
      input:
        ret935.1: {}
      output:
        '3927': {}
      attr:
        start: 2
        end: 2
    aten::add_3145:
      type: Add
      input:
        '3927': {}
        '3916': {}
      output:
        '3932': {}
    aten::slice_2688:
      type: Slice
      input:
        '3923': {}
        '3916': {}
        '3932': {}
      output:
        '3934': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2144:
      type: Slice
      input:
        '3934': {}
      output:
        '3935': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2689:
      type: Slice
      input:
        '3925': {}
        '3916': {}
        '3932': {}
      output:
        '3936': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2145:
      type: Slice
      input:
        '3936': {}
      output:
        '3937': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4853:
      type: Mul
      input:
        ret935.1: {}
        '3935': {}
      output:
        ret951.1: {}
      attr:
        algorithm: mul
    aten::size_2146:
      type: Shape
      input:
        ret935.1: {}
      output:
        '3941': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_198:
      type: Div
      input:
        '3941': {}
        '495': {}
      output:
        ret953.1: {}
      attr:
        algorithm: div
    aten::slice_2147:
      type: Slice
      input:
        ret935.1: {}
        ret953.1: {}
      output:
        ret954.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2148:
      type: Slice
      input:
        ret935.1: {}
        ret953.1: {}
      output:
        ret955.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4860:
      type: Neg
      input:
        ret955.1: {}
        aten::neg_4860_mul_val: {}
      output:
        ret956.1: {}
      attr:
        algorithm: mul
    aten::cat_2475:
      type: Concat
      input:
        ret956.1: {}
        ret954.1: {}
      output:
        ret957.1: {}
      attr:
        axis: -1
    aten::mul_4857:
      type: Mul
      input:
        ret957.1: {}
        '3937': {}
      output:
        ret958.1: {}
      attr:
        algorithm: mul
    aten::add_3146:
      type: Add
      input:
        ret951.1: {}
        ret958.1: {}
      output:
        args43.1: {}
    aten::mul_4855:
      type: Mul
      input:
        ret941.1: {}
        '3935': {}
      output:
        ret959.1: {}
      attr:
        algorithm: mul
    aten::size_2149:
      type: Shape
      input:
        ret941.1: {}
      output:
        '3963': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_199:
      type: Div
      input:
        '3963': {}
        '495': {}
      output:
        ret961.1: {}
      attr:
        algorithm: div
    aten::slice_2150:
      type: Slice
      input:
        ret941.1: {}
        ret961.1: {}
      output:
        ret962.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2151:
      type: Slice
      input:
        ret941.1: {}
        ret961.1: {}
      output:
        ret963.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4862:
      type: Neg
      input:
        ret963.1: {}
        aten::neg_4862_mul_val: {}
      output:
        ret964.1: {}
      attr:
        algorithm: mul
    aten::cat_2476:
      type: Concat
      input:
        ret964.1: {}
        ret962.1: {}
      output:
        ret965.1: {}
      attr:
        axis: -1
    aten::mul_4858:
      type: Mul
      input:
        ret965.1: {}
        '3937': {}
      output:
        ret966.1: {}
      attr:
        algorithm: mul
    aten::add_3147:
      type: Add
      input:
        ret959.1: {}
        ret966.1: {}
      output:
        '3981': {}
    aten::cat_2690:
      type: Concat
      input:
        x21.1: {}
        '3981': {}
      output:
        ret967.1: {}
      attr:
        axis: 2
    aten::cat_2691:
      type: Concat
      input:
        x22.1: {}
        ret947.1: {}
      output:
        ret968.1: {}
      attr:
        axis: 2
    aten::transpose_2152:
      type: Reorder
      input:
        ret967.1: {}
      output:
        ret969.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4865:
      type: Matmul
      input:
        args43.1: {}
        ret969.1: {}
      output:
        ret972.1: {}
    aten::div_267:
      type: Div
      input:
        ret972.1: {}
        '496': {}
      output:
        ret973.1: {}
      attr:
        algorithm: div
    aten::add_3148:
      type: Add
      input:
        ret973.1: {}
        attention_mask0.1: {}
      output:
        attn_weights22.1: {}
    aten::max_307:
      type: Max
      input:
        attn_weights22.1: {}
        '497': {}
      output:
        input22.1: {}
    aten::softmax_2420:
      type: Softmax
      input:
        input22.1: {}
      output:
        '4003': {}
      attr:
        axis: -1
    aten::matmul_4868:
      type: Matmul
      input:
        '4003': {}
        ret968.1: {}
      output:
        ret976.1: {}
    aten::transpose_2692:
      type: Reorder
      input:
        ret976.1: {}
      output:
        ret977.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4870:
      type: Reshape
      input:
        ret977.1: {}
        '3865': {}
        '3867': {}
      output:
        ret978.1: {}
    aten::mul_512:
      type: Mul
      input:
        ret978.1: {}
        '674': {}
      output:
        ret979.1: {}
      attr:
        algorithm: mul
    aten::linear_3516:
      type: InnerProduct
      input:
        ret979.1: {}
        '675': {}
        aten::linear_3516_bias: {}
      output:
        ret982.1: {}
    aten::add_3149:
      type: Add
      input:
        ret921.1: {}
        ret982.1: {}
      output:
        ret983.1: {}
    aten::pow_2693:
      type: Pow
      input:
        ret983.1: {}
        aten::pow_2693_other: {}
      output:
        ret984.1: {}
    aten::mean_996:
      type: ReduceMean
      input:
        ret984.1: {}
      output:
        ret985.1: {}
    aten::add_31:
      type: Add
      input:
        ret985.1: {}
        '485': {}
      output:
        ret986.1: {}
    aten::rsqrt_4873:
      type: Rsqrt
      input:
        ret986.1: {}
      output:
        ret987.1: {}
    aten::mul_4872:
      type: Mul
      input:
        ret983.1: {}
        ret987.1: {}
      output:
        ret988.1: {}
      attr:
        algorithm: mul
    aten::mul_514:
      type: Mul
      input:
        '676': {}
        ret988.1: {}
      output:
        ret989.1: {}
      attr:
        algorithm: mul
    aten::mul_515:
      type: Mul
      input:
        ret989.1: {}
        '677': {}
      output:
        ret990.1: {}
      attr:
        algorithm: mul
    aten::linear_3517:
      type: InnerProduct
      input:
        ret990.1: {}
        '678': {}
        aten::linear_3517_bias: {}
      output:
        ret993.1: {}
    aten::silu_4875:
      type: Swish
      input:
        ret993.1: {}
      output:
        ret994.1: {}
    aten::mul_517:
      type: Mul
      input:
        ret989.1: {}
        '679': {}
      output:
        ret995.1: {}
      attr:
        algorithm: mul
    aten::linear_3518:
      type: InnerProduct
      input:
        ret995.1: {}
        '680': {}
        aten::linear_3518_bias: {}
      output:
        ret998.1: {}
    aten::mul_4876:
      type: Mul
      input:
        ret994.1: {}
        ret998.1: {}
      output:
        ret999.1: {}
      attr:
        algorithm: mul
    aten::mul_519:
      type: Mul
      input:
        ret999.1: {}
        '681': {}
      output:
        ret1000.1: {}
      attr:
        algorithm: mul
    aten::linear_3519:
      type: InnerProduct
      input:
        ret1000.1: {}
        '682': {}
        aten::linear_3519_bias: {}
      output:
        ret1003.1: {}
    aten::add_3150:
      type: Add
      input:
        ret983.1: {}
        ret1003.1: {}
      output:
        ret1004.1: {}
    aten::pow_2694:
      type: Pow
      input:
        ret1004.1: {}
        aten::pow_2694_other: {}
      output:
        ret1005.1: {}
    aten::mean_997:
      type: ReduceMean
      input:
        ret1005.1: {}
      output:
        ret1006.1: {}
    aten::add_32:
      type: Add
      input:
        ret1006.1: {}
        '485': {}
      output:
        ret1007.1: {}
    aten::rsqrt_4880:
      type: Rsqrt
      input:
        ret1007.1: {}
      output:
        ret1008.1: {}
    aten::mul_4879:
      type: Mul
      input:
        ret1004.1: {}
        ret1008.1: {}
      output:
        ret1009.1: {}
      attr:
        algorithm: mul
    aten::mul_521:
      type: Mul
      input:
        '683': {}
        ret1009.1: {}
      output:
        ret1010.1: {}
      attr:
        algorithm: mul
    aten::size_3401:
      type: Shape
      input:
        ret1010.1: {}
      output:
        '4088': {}
      attr:
        start: 0
        end: 0
    aten::size_3151:
      type: Shape
      input:
        ret1010.1: {}
      output:
        '4090': {}
      attr:
        start: 1
        end: 1
    aten::mul_522:
      type: Mul
      input:
        ret1010.1: {}
        '684': {}
      output:
        ret1013.1: {}
      attr:
        algorithm: mul
    aten::linear_3520:
      type: InnerProduct
      input:
        ret1013.1: {}
        '685': {}
        aten::linear_3520_bias: {}
      output:
        ret1016.1: {}
    aten::view_4882:
      type: View
      input:
        ret1016.1: {}
        '4088': {}
        '4090': {}
      output:
        ret1017.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2695:
      type: Reorder
      input:
        ret1017.1: {}
      output:
        ret1018.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_524:
      type: Mul
      input:
        ret1010.1: {}
        '686': {}
      output:
        ret1019.1: {}
      attr:
        algorithm: mul
    aten::linear_3521:
      type: InnerProduct
      input:
        ret1019.1: {}
        '687': {}
        aten::linear_3521_bias: {}
      output:
        ret1022.1: {}
    aten::view_4883:
      type: View
      input:
        ret1022.1: {}
        '4088': {}
        '4090': {}
      output:
        ret1023.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2696:
      type: Reorder
      input:
        ret1023.1: {}
      output:
        ret1024.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_526:
      type: Mul
      input:
        ret1010.1: {}
        '688': {}
      output:
        ret1025.1: {}
      attr:
        algorithm: mul
    aten::linear_3522:
      type: InnerProduct
      input:
        ret1025.1: {}
        '689': {}
        aten::linear_3522_bias: {}
      output:
        ret1028.1: {}
    aten::view_4884:
      type: View
      input:
        ret1028.1: {}
        '4088': {}
        '4090': {}
      output:
        ret1029.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2697:
      type: Reorder
      input:
        ret1029.1: {}
      output:
        ret1030.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2698:
      type: Shape
      input:
        ret1024.1: {}
      output:
        '4136': {}
      attr:
        start: 2
        end: 2
    aten::size_2699:
      type: Shape
      input:
        x23.1: {}
      output:
        '4139': {}
      attr:
        start: 2
        end: 2
    aten::add_3359:
      type: Add
      input:
        '4136': {}
        '4139': {}
      output:
        seq_len11.1: {}
    aten::slice_108:
      type: Slice
      input:
        '493': {}
        seq_len11.1: {}
      output:
        '4146': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_148:
      type: Slice
      input:
        '494': {}
        seq_len11.1: {}
      output:
        '4148': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2700:
      type: Shape
      input:
        ret1018.1: {}
      output:
        '4150': {}
      attr:
        start: 2
        end: 2
    aten::add_3152:
      type: Add
      input:
        '4150': {}
        '4139': {}
      output:
        '4155': {}
    aten::slice_2701:
      type: Slice
      input:
        '4146': {}
        '4139': {}
        '4155': {}
      output:
        '4157': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2153:
      type: Slice
      input:
        '4157': {}
      output:
        '4158': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2702:
      type: Slice
      input:
        '4148': {}
        '4139': {}
        '4155': {}
      output:
        '4159': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2154:
      type: Slice
      input:
        '4159': {}
      output:
        '4160': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4885:
      type: Mul
      input:
        ret1018.1: {}
        '4158': {}
      output:
        ret1034.1: {}
      attr:
        algorithm: mul
    aten::size_2155:
      type: Shape
      input:
        ret1018.1: {}
      output:
        '4164': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_200:
      type: Div
      input:
        '4164': {}
        '495': {}
      output:
        ret1036.1: {}
      attr:
        algorithm: div
    aten::slice_2156:
      type: Slice
      input:
        ret1018.1: {}
        ret1036.1: {}
      output:
        ret1037.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2157:
      type: Slice
      input:
        ret1018.1: {}
        ret1036.1: {}
      output:
        ret1038.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4892:
      type: Neg
      input:
        ret1038.1: {}
        aten::neg_4892_mul_val: {}
      output:
        ret1039.1: {}
      attr:
        algorithm: mul
    aten::cat_2477:
      type: Concat
      input:
        ret1039.1: {}
        ret1037.1: {}
      output:
        ret1040.1: {}
      attr:
        axis: -1
    aten::mul_4889:
      type: Mul
      input:
        ret1040.1: {}
        '4160': {}
      output:
        ret1041.1: {}
      attr:
        algorithm: mul
    aten::add_3153:
      type: Add
      input:
        ret1034.1: {}
        ret1041.1: {}
      output:
        args47.1: {}
    aten::mul_4887:
      type: Mul
      input:
        ret1024.1: {}
        '4158': {}
      output:
        ret1042.1: {}
      attr:
        algorithm: mul
    aten::size_2158:
      type: Shape
      input:
        ret1024.1: {}
      output:
        '4186': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_201:
      type: Div
      input:
        '4186': {}
        '495': {}
      output:
        ret1044.1: {}
      attr:
        algorithm: div
    aten::slice_2159:
      type: Slice
      input:
        ret1024.1: {}
        ret1044.1: {}
      output:
        ret1045.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2160:
      type: Slice
      input:
        ret1024.1: {}
        ret1044.1: {}
      output:
        ret1046.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4894:
      type: Neg
      input:
        ret1046.1: {}
        aten::neg_4894_mul_val: {}
      output:
        ret1047.1: {}
      attr:
        algorithm: mul
    aten::cat_2478:
      type: Concat
      input:
        ret1047.1: {}
        ret1045.1: {}
      output:
        ret1048.1: {}
      attr:
        axis: -1
    aten::mul_4890:
      type: Mul
      input:
        ret1048.1: {}
        '4160': {}
      output:
        ret1049.1: {}
      attr:
        algorithm: mul
    aten::add_3154:
      type: Add
      input:
        ret1042.1: {}
        ret1049.1: {}
      output:
        '4204': {}
    aten::cat_2703:
      type: Concat
      input:
        x23.1: {}
        '4204': {}
      output:
        ret1050.1: {}
      attr:
        axis: 2
    aten::cat_2704:
      type: Concat
      input:
        x24.1: {}
        ret1030.1: {}
      output:
        ret1051.1: {}
      attr:
        axis: 2
    aten::transpose_2161:
      type: Reorder
      input:
        ret1050.1: {}
      output:
        ret1052.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4897:
      type: Matmul
      input:
        args47.1: {}
        ret1052.1: {}
      output:
        ret1055.1: {}
    aten::div_268:
      type: Div
      input:
        ret1055.1: {}
        '496': {}
      output:
        ret1056.1: {}
      attr:
        algorithm: div
    aten::add_3155:
      type: Add
      input:
        ret1056.1: {}
        attention_mask0.1: {}
      output:
        attn_weights24.1: {}
    aten::max_308:
      type: Max
      input:
        attn_weights24.1: {}
        '497': {}
      output:
        input24.1: {}
    aten::softmax_2421:
      type: Softmax
      input:
        input24.1: {}
      output:
        '4226': {}
      attr:
        axis: -1
    aten::matmul_4900:
      type: Matmul
      input:
        '4226': {}
        ret1051.1: {}
      output:
        ret1059.1: {}
    aten::transpose_2705:
      type: Reorder
      input:
        ret1059.1: {}
      output:
        ret1060.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4902:
      type: Reshape
      input:
        ret1060.1: {}
        '4088': {}
        '4090': {}
      output:
        ret1061.1: {}
    aten::mul_528:
      type: Mul
      input:
        ret1061.1: {}
        '690': {}
      output:
        ret1062.1: {}
      attr:
        algorithm: mul
    aten::linear_3523:
      type: InnerProduct
      input:
        ret1062.1: {}
        '691': {}
        aten::linear_3523_bias: {}
      output:
        ret1065.1: {}
    aten::add_3156:
      type: Add
      input:
        ret1004.1: {}
        ret1065.1: {}
      output:
        ret1066.1: {}
    aten::pow_2706:
      type: Pow
      input:
        ret1066.1: {}
        aten::pow_2706_other: {}
      output:
        ret1067.1: {}
    aten::mean_998:
      type: ReduceMean
      input:
        ret1067.1: {}
      output:
        ret1068.1: {}
    aten::add_33:
      type: Add
      input:
        ret1068.1: {}
        '485': {}
      output:
        ret1069.1: {}
    aten::rsqrt_4905:
      type: Rsqrt
      input:
        ret1069.1: {}
      output:
        ret1070.1: {}
    aten::mul_4904:
      type: Mul
      input:
        ret1066.1: {}
        ret1070.1: {}
      output:
        ret1071.1: {}
      attr:
        algorithm: mul
    aten::mul_530:
      type: Mul
      input:
        '692': {}
        ret1071.1: {}
      output:
        ret1072.1: {}
      attr:
        algorithm: mul
    aten::mul_531:
      type: Mul
      input:
        ret1072.1: {}
        '693': {}
      output:
        ret1073.1: {}
      attr:
        algorithm: mul
    aten::linear_3524:
      type: InnerProduct
      input:
        ret1073.1: {}
        '694': {}
        aten::linear_3524_bias: {}
      output:
        ret1076.1: {}
    aten::silu_4907:
      type: Swish
      input:
        ret1076.1: {}
      output:
        ret1077.1: {}
    aten::mul_533:
      type: Mul
      input:
        ret1072.1: {}
        '695': {}
      output:
        ret1078.1: {}
      attr:
        algorithm: mul
    aten::linear_3525:
      type: InnerProduct
      input:
        ret1078.1: {}
        '696': {}
        aten::linear_3525_bias: {}
      output:
        ret1081.1: {}
    aten::mul_4908:
      type: Mul
      input:
        ret1077.1: {}
        ret1081.1: {}
      output:
        ret1082.1: {}
      attr:
        algorithm: mul
    aten::mul_535:
      type: Mul
      input:
        ret1082.1: {}
        '697': {}
      output:
        ret1083.1: {}
      attr:
        algorithm: mul
    aten::linear_3526:
      type: InnerProduct
      input:
        ret1083.1: {}
        '698': {}
        aten::linear_3526_bias: {}
      output:
        ret1086.1: {}
    aten::add_3157:
      type: Add
      input:
        ret1066.1: {}
        ret1086.1: {}
      output:
        ret1087.1: {}
    aten::pow_2707:
      type: Pow
      input:
        ret1087.1: {}
        aten::pow_2707_other: {}
      output:
        ret1088.1: {}
    aten::mean_999:
      type: ReduceMean
      input:
        ret1088.1: {}
      output:
        ret1089.1: {}
    aten::add_34:
      type: Add
      input:
        ret1089.1: {}
        '485': {}
      output:
        ret1090.1: {}
    aten::rsqrt_4912:
      type: Rsqrt
      input:
        ret1090.1: {}
      output:
        ret1091.1: {}
    aten::mul_4911:
      type: Mul
      input:
        ret1087.1: {}
        ret1091.1: {}
      output:
        ret1092.1: {}
      attr:
        algorithm: mul
    aten::mul_537:
      type: Mul
      input:
        '699': {}
        ret1092.1: {}
      output:
        ret1093.1: {}
      attr:
        algorithm: mul
    aten::size_3402:
      type: Shape
      input:
        ret1093.1: {}
      output:
        '4311': {}
      attr:
        start: 0
        end: 0
    aten::size_3158:
      type: Shape
      input:
        ret1093.1: {}
      output:
        '4313': {}
      attr:
        start: 1
        end: 1
    aten::mul_538:
      type: Mul
      input:
        ret1093.1: {}
        '700': {}
      output:
        ret1096.1: {}
      attr:
        algorithm: mul
    aten::linear_3527:
      type: InnerProduct
      input:
        ret1096.1: {}
        '701': {}
        aten::linear_3527_bias: {}
      output:
        ret1099.1: {}
    aten::view_4914:
      type: View
      input:
        ret1099.1: {}
        '4311': {}
        '4313': {}
      output:
        ret1100.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2708:
      type: Reorder
      input:
        ret1100.1: {}
      output:
        ret1101.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_540:
      type: Mul
      input:
        ret1093.1: {}
        '702': {}
      output:
        ret1102.1: {}
      attr:
        algorithm: mul
    aten::linear_3528:
      type: InnerProduct
      input:
        ret1102.1: {}
        '703': {}
        aten::linear_3528_bias: {}
      output:
        ret1105.1: {}
    aten::view_4915:
      type: View
      input:
        ret1105.1: {}
        '4311': {}
        '4313': {}
      output:
        ret1106.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2709:
      type: Reorder
      input:
        ret1106.1: {}
      output:
        ret1107.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_542:
      type: Mul
      input:
        ret1093.1: {}
        '704': {}
      output:
        ret1108.1: {}
      attr:
        algorithm: mul
    aten::linear_3529:
      type: InnerProduct
      input:
        ret1108.1: {}
        '705': {}
        aten::linear_3529_bias: {}
      output:
        ret1111.1: {}
    aten::view_4916:
      type: View
      input:
        ret1111.1: {}
        '4311': {}
        '4313': {}
      output:
        ret1112.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2710:
      type: Reorder
      input:
        ret1112.1: {}
      output:
        ret1113.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2711:
      type: Shape
      input:
        ret1107.1: {}
      output:
        '4359': {}
      attr:
        start: 2
        end: 2
    aten::size_2712:
      type: Shape
      input:
        x25.1: {}
      output:
        '4362': {}
      attr:
        start: 2
        end: 2
    aten::add_3360:
      type: Add
      input:
        '4359': {}
        '4362': {}
      output:
        seq_len12.1: {}
    aten::slice_109:
      type: Slice
      input:
        '493': {}
        seq_len12.1: {}
      output:
        '4369': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_149:
      type: Slice
      input:
        '494': {}
        seq_len12.1: {}
      output:
        '4371': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2713:
      type: Shape
      input:
        ret1101.1: {}
      output:
        '4373': {}
      attr:
        start: 2
        end: 2
    aten::add_3159:
      type: Add
      input:
        '4373': {}
        '4362': {}
      output:
        '4378': {}
    aten::slice_2714:
      type: Slice
      input:
        '4369': {}
        '4362': {}
        '4378': {}
      output:
        '4380': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2162:
      type: Slice
      input:
        '4380': {}
      output:
        '4381': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2715:
      type: Slice
      input:
        '4371': {}
        '4362': {}
        '4378': {}
      output:
        '4382': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2163:
      type: Slice
      input:
        '4382': {}
      output:
        '4383': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4917:
      type: Mul
      input:
        ret1101.1: {}
        '4381': {}
      output:
        ret1117.1: {}
      attr:
        algorithm: mul
    aten::size_2164:
      type: Shape
      input:
        ret1101.1: {}
      output:
        '4387': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_202:
      type: Div
      input:
        '4387': {}
        '495': {}
      output:
        ret1119.1: {}
      attr:
        algorithm: div
    aten::slice_2165:
      type: Slice
      input:
        ret1101.1: {}
        ret1119.1: {}
      output:
        ret1120.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2166:
      type: Slice
      input:
        ret1101.1: {}
        ret1119.1: {}
      output:
        ret1121.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4924:
      type: Neg
      input:
        ret1121.1: {}
        aten::neg_4924_mul_val: {}
      output:
        ret1122.1: {}
      attr:
        algorithm: mul
    aten::cat_2479:
      type: Concat
      input:
        ret1122.1: {}
        ret1120.1: {}
      output:
        ret1123.1: {}
      attr:
        axis: -1
    aten::mul_4921:
      type: Mul
      input:
        ret1123.1: {}
        '4383': {}
      output:
        ret1124.1: {}
      attr:
        algorithm: mul
    aten::add_3160:
      type: Add
      input:
        ret1117.1: {}
        ret1124.1: {}
      output:
        args51.1: {}
    aten::mul_4919:
      type: Mul
      input:
        ret1107.1: {}
        '4381': {}
      output:
        ret1125.1: {}
      attr:
        algorithm: mul
    aten::size_2167:
      type: Shape
      input:
        ret1107.1: {}
      output:
        '4409': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_203:
      type: Div
      input:
        '4409': {}
        '495': {}
      output:
        ret1127.1: {}
      attr:
        algorithm: div
    aten::slice_2168:
      type: Slice
      input:
        ret1107.1: {}
        ret1127.1: {}
      output:
        ret1128.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2169:
      type: Slice
      input:
        ret1107.1: {}
        ret1127.1: {}
      output:
        ret1129.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4926:
      type: Neg
      input:
        ret1129.1: {}
        aten::neg_4926_mul_val: {}
      output:
        ret1130.1: {}
      attr:
        algorithm: mul
    aten::cat_2480:
      type: Concat
      input:
        ret1130.1: {}
        ret1128.1: {}
      output:
        ret1131.1: {}
      attr:
        axis: -1
    aten::mul_4922:
      type: Mul
      input:
        ret1131.1: {}
        '4383': {}
      output:
        ret1132.1: {}
      attr:
        algorithm: mul
    aten::add_3161:
      type: Add
      input:
        ret1125.1: {}
        ret1132.1: {}
      output:
        '4427': {}
    aten::cat_2716:
      type: Concat
      input:
        x25.1: {}
        '4427': {}
      output:
        ret1133.1: {}
      attr:
        axis: 2
    aten::cat_2717:
      type: Concat
      input:
        x26.1: {}
        ret1113.1: {}
      output:
        ret1134.1: {}
      attr:
        axis: 2
    aten::transpose_2170:
      type: Reorder
      input:
        ret1133.1: {}
      output:
        ret1135.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4929:
      type: Matmul
      input:
        args51.1: {}
        ret1135.1: {}
      output:
        ret1138.1: {}
    aten::div_269:
      type: Div
      input:
        ret1138.1: {}
        '496': {}
      output:
        ret1139.1: {}
      attr:
        algorithm: div
    aten::add_3162:
      type: Add
      input:
        ret1139.1: {}
        attention_mask0.1: {}
      output:
        attn_weights26.1: {}
    aten::max_309:
      type: Max
      input:
        attn_weights26.1: {}
        '497': {}
      output:
        input26.1: {}
    aten::softmax_2422:
      type: Softmax
      input:
        input26.1: {}
      output:
        '4449': {}
      attr:
        axis: -1
    aten::matmul_4932:
      type: Matmul
      input:
        '4449': {}
        ret1134.1: {}
      output:
        ret1142.1: {}
    aten::transpose_2718:
      type: Reorder
      input:
        ret1142.1: {}
      output:
        ret1143.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4934:
      type: Reshape
      input:
        ret1143.1: {}
        '4311': {}
        '4313': {}
      output:
        ret1144.1: {}
    aten::mul_544:
      type: Mul
      input:
        ret1144.1: {}
        '706': {}
      output:
        ret1145.1: {}
      attr:
        algorithm: mul
    aten::linear_3530:
      type: InnerProduct
      input:
        ret1145.1: {}
        '707': {}
        aten::linear_3530_bias: {}
      output:
        ret1148.1: {}
    aten::add_3163:
      type: Add
      input:
        ret1087.1: {}
        ret1148.1: {}
      output:
        ret1149.1: {}
    aten::pow_2719:
      type: Pow
      input:
        ret1149.1: {}
        aten::pow_2719_other: {}
      output:
        ret1150.1: {}
    aten::mean_1000:
      type: ReduceMean
      input:
        ret1150.1: {}
      output:
        ret1151.1: {}
    aten::add_35:
      type: Add
      input:
        ret1151.1: {}
        '485': {}
      output:
        ret1152.1: {}
    aten::rsqrt_4937:
      type: Rsqrt
      input:
        ret1152.1: {}
      output:
        ret1153.1: {}
    aten::mul_4936:
      type: Mul
      input:
        ret1149.1: {}
        ret1153.1: {}
      output:
        ret1154.1: {}
      attr:
        algorithm: mul
    aten::mul_546:
      type: Mul
      input:
        '708': {}
        ret1154.1: {}
      output:
        ret1155.1: {}
      attr:
        algorithm: mul
    aten::mul_547:
      type: Mul
      input:
        ret1155.1: {}
        '709': {}
      output:
        ret1156.1: {}
      attr:
        algorithm: mul
    aten::linear_3531:
      type: InnerProduct
      input:
        ret1156.1: {}
        '710': {}
        aten::linear_3531_bias: {}
      output:
        ret1159.1: {}
    aten::silu_4939:
      type: Swish
      input:
        ret1159.1: {}
      output:
        ret1160.1: {}
    aten::mul_549:
      type: Mul
      input:
        ret1155.1: {}
        '711': {}
      output:
        ret1161.1: {}
      attr:
        algorithm: mul
    aten::linear_3532:
      type: InnerProduct
      input:
        ret1161.1: {}
        '712': {}
        aten::linear_3532_bias: {}
      output:
        ret1164.1: {}
    aten::mul_4940:
      type: Mul
      input:
        ret1160.1: {}
        ret1164.1: {}
      output:
        ret1165.1: {}
      attr:
        algorithm: mul
    aten::mul_551:
      type: Mul
      input:
        ret1165.1: {}
        '713': {}
      output:
        ret1166.1: {}
      attr:
        algorithm: mul
    aten::linear_3533:
      type: InnerProduct
      input:
        ret1166.1: {}
        '714': {}
        aten::linear_3533_bias: {}
      output:
        ret1169.1: {}
    aten::add_3164:
      type: Add
      input:
        ret1149.1: {}
        ret1169.1: {}
      output:
        ret1170.1: {}
    aten::pow_2720:
      type: Pow
      input:
        ret1170.1: {}
        aten::pow_2720_other: {}
      output:
        ret1171.1: {}
    aten::mean_1001:
      type: ReduceMean
      input:
        ret1171.1: {}
      output:
        ret1172.1: {}
    aten::add_36:
      type: Add
      input:
        ret1172.1: {}
        '485': {}
      output:
        ret1173.1: {}
    aten::rsqrt_4944:
      type: Rsqrt
      input:
        ret1173.1: {}
      output:
        ret1174.1: {}
    aten::mul_4943:
      type: Mul
      input:
        ret1170.1: {}
        ret1174.1: {}
      output:
        ret1175.1: {}
      attr:
        algorithm: mul
    aten::mul_553:
      type: Mul
      input:
        '715': {}
        ret1175.1: {}
      output:
        ret1176.1: {}
      attr:
        algorithm: mul
    aten::size_3403:
      type: Shape
      input:
        ret1176.1: {}
      output:
        '4534': {}
      attr:
        start: 0
        end: 0
    aten::size_3165:
      type: Shape
      input:
        ret1176.1: {}
      output:
        '4536': {}
      attr:
        start: 1
        end: 1
    aten::mul_554:
      type: Mul
      input:
        ret1176.1: {}
        '716': {}
      output:
        ret1179.1: {}
      attr:
        algorithm: mul
    aten::linear_3534:
      type: InnerProduct
      input:
        ret1179.1: {}
        '717': {}
        aten::linear_3534_bias: {}
      output:
        ret1182.1: {}
    aten::view_4946:
      type: View
      input:
        ret1182.1: {}
        '4534': {}
        '4536': {}
      output:
        ret1183.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2721:
      type: Reorder
      input:
        ret1183.1: {}
      output:
        ret1184.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_556:
      type: Mul
      input:
        ret1176.1: {}
        '718': {}
      output:
        ret1185.1: {}
      attr:
        algorithm: mul
    aten::linear_3535:
      type: InnerProduct
      input:
        ret1185.1: {}
        '719': {}
        aten::linear_3535_bias: {}
      output:
        ret1188.1: {}
    aten::view_4947:
      type: View
      input:
        ret1188.1: {}
        '4534': {}
        '4536': {}
      output:
        ret1189.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2722:
      type: Reorder
      input:
        ret1189.1: {}
      output:
        ret1190.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_558:
      type: Mul
      input:
        ret1176.1: {}
        '720': {}
      output:
        ret1191.1: {}
      attr:
        algorithm: mul
    aten::linear_3536:
      type: InnerProduct
      input:
        ret1191.1: {}
        '721': {}
        aten::linear_3536_bias: {}
      output:
        ret1194.1: {}
    aten::view_4948:
      type: View
      input:
        ret1194.1: {}
        '4534': {}
        '4536': {}
      output:
        ret1195.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2723:
      type: Reorder
      input:
        ret1195.1: {}
      output:
        ret1196.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2724:
      type: Shape
      input:
        ret1190.1: {}
      output:
        '4582': {}
      attr:
        start: 2
        end: 2
    aten::size_2725:
      type: Shape
      input:
        x27.1: {}
      output:
        '4585': {}
      attr:
        start: 2
        end: 2
    aten::add_3361:
      type: Add
      input:
        '4582': {}
        '4585': {}
      output:
        seq_len13.1: {}
    aten::slice_110:
      type: Slice
      input:
        '493': {}
        seq_len13.1: {}
      output:
        '4592': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_150:
      type: Slice
      input:
        '494': {}
        seq_len13.1: {}
      output:
        '4594': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2726:
      type: Shape
      input:
        ret1184.1: {}
      output:
        '4596': {}
      attr:
        start: 2
        end: 2
    aten::add_3166:
      type: Add
      input:
        '4596': {}
        '4585': {}
      output:
        '4601': {}
    aten::slice_2727:
      type: Slice
      input:
        '4592': {}
        '4585': {}
        '4601': {}
      output:
        '4603': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2171:
      type: Slice
      input:
        '4603': {}
      output:
        '4604': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2728:
      type: Slice
      input:
        '4594': {}
        '4585': {}
        '4601': {}
      output:
        '4605': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2172:
      type: Slice
      input:
        '4605': {}
      output:
        '4606': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4949:
      type: Mul
      input:
        ret1184.1: {}
        '4604': {}
      output:
        ret1200.1: {}
      attr:
        algorithm: mul
    aten::size_2173:
      type: Shape
      input:
        ret1184.1: {}
      output:
        '4610': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_204:
      type: Div
      input:
        '4610': {}
        '495': {}
      output:
        ret1202.1: {}
      attr:
        algorithm: div
    aten::slice_2174:
      type: Slice
      input:
        ret1184.1: {}
        ret1202.1: {}
      output:
        ret1203.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2175:
      type: Slice
      input:
        ret1184.1: {}
        ret1202.1: {}
      output:
        ret1204.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4956:
      type: Neg
      input:
        ret1204.1: {}
        aten::neg_4956_mul_val: {}
      output:
        ret1205.1: {}
      attr:
        algorithm: mul
    aten::cat_2481:
      type: Concat
      input:
        ret1205.1: {}
        ret1203.1: {}
      output:
        ret1206.1: {}
      attr:
        axis: -1
    aten::mul_4953:
      type: Mul
      input:
        ret1206.1: {}
        '4606': {}
      output:
        ret1207.1: {}
      attr:
        algorithm: mul
    aten::add_3167:
      type: Add
      input:
        ret1200.1: {}
        ret1207.1: {}
      output:
        args55.1: {}
    aten::mul_4951:
      type: Mul
      input:
        ret1190.1: {}
        '4604': {}
      output:
        ret1208.1: {}
      attr:
        algorithm: mul
    aten::size_2176:
      type: Shape
      input:
        ret1190.1: {}
      output:
        '4632': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_205:
      type: Div
      input:
        '4632': {}
        '495': {}
      output:
        ret1210.1: {}
      attr:
        algorithm: div
    aten::slice_2177:
      type: Slice
      input:
        ret1190.1: {}
        ret1210.1: {}
      output:
        ret1211.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2178:
      type: Slice
      input:
        ret1190.1: {}
        ret1210.1: {}
      output:
        ret1212.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4958:
      type: Neg
      input:
        ret1212.1: {}
        aten::neg_4958_mul_val: {}
      output:
        ret1213.1: {}
      attr:
        algorithm: mul
    aten::cat_2482:
      type: Concat
      input:
        ret1213.1: {}
        ret1211.1: {}
      output:
        ret1214.1: {}
      attr:
        axis: -1
    aten::mul_4954:
      type: Mul
      input:
        ret1214.1: {}
        '4606': {}
      output:
        ret1215.1: {}
      attr:
        algorithm: mul
    aten::add_3168:
      type: Add
      input:
        ret1208.1: {}
        ret1215.1: {}
      output:
        '4650': {}
    aten::cat_2729:
      type: Concat
      input:
        x27.1: {}
        '4650': {}
      output:
        ret1216.1: {}
      attr:
        axis: 2
    aten::cat_2730:
      type: Concat
      input:
        x28.1: {}
        ret1196.1: {}
      output:
        ret1217.1: {}
      attr:
        axis: 2
    aten::transpose_2179:
      type: Reorder
      input:
        ret1216.1: {}
      output:
        ret1218.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4961:
      type: Matmul
      input:
        args55.1: {}
        ret1218.1: {}
      output:
        ret1221.1: {}
    aten::div_270:
      type: Div
      input:
        ret1221.1: {}
        '496': {}
      output:
        ret1222.1: {}
      attr:
        algorithm: div
    aten::add_3169:
      type: Add
      input:
        ret1222.1: {}
        attention_mask0.1: {}
      output:
        attn_weights28.1: {}
    aten::max_310:
      type: Max
      input:
        attn_weights28.1: {}
        '497': {}
      output:
        input28.1: {}
    aten::softmax_2423:
      type: Softmax
      input:
        input28.1: {}
      output:
        '4672': {}
      attr:
        axis: -1
    aten::matmul_4964:
      type: Matmul
      input:
        '4672': {}
        ret1217.1: {}
      output:
        ret1225.1: {}
    aten::transpose_2731:
      type: Reorder
      input:
        ret1225.1: {}
      output:
        ret1226.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4966:
      type: Reshape
      input:
        ret1226.1: {}
        '4534': {}
        '4536': {}
      output:
        ret1227.1: {}
    aten::mul_560:
      type: Mul
      input:
        ret1227.1: {}
        '722': {}
      output:
        ret1228.1: {}
      attr:
        algorithm: mul
    aten::linear_3537:
      type: InnerProduct
      input:
        ret1228.1: {}
        '723': {}
        aten::linear_3537_bias: {}
      output:
        ret1231.1: {}
    aten::add_3170:
      type: Add
      input:
        ret1170.1: {}
        ret1231.1: {}
      output:
        ret1232.1: {}
    aten::pow_2732:
      type: Pow
      input:
        ret1232.1: {}
        aten::pow_2732_other: {}
      output:
        ret1233.1: {}
    aten::mean_1002:
      type: ReduceMean
      input:
        ret1233.1: {}
      output:
        ret1234.1: {}
    aten::add_37:
      type: Add
      input:
        ret1234.1: {}
        '485': {}
      output:
        ret1235.1: {}
    aten::rsqrt_4969:
      type: Rsqrt
      input:
        ret1235.1: {}
      output:
        ret1236.1: {}
    aten::mul_4968:
      type: Mul
      input:
        ret1232.1: {}
        ret1236.1: {}
      output:
        ret1237.1: {}
      attr:
        algorithm: mul
    aten::mul_562:
      type: Mul
      input:
        '724': {}
        ret1237.1: {}
      output:
        ret1238.1: {}
      attr:
        algorithm: mul
    aten::mul_563:
      type: Mul
      input:
        ret1238.1: {}
        '725': {}
      output:
        ret1239.1: {}
      attr:
        algorithm: mul
    aten::linear_3538:
      type: InnerProduct
      input:
        ret1239.1: {}
        '726': {}
        aten::linear_3538_bias: {}
      output:
        ret1242.1: {}
    aten::silu_4971:
      type: Swish
      input:
        ret1242.1: {}
      output:
        ret1243.1: {}
    aten::mul_565:
      type: Mul
      input:
        ret1238.1: {}
        '727': {}
      output:
        ret1244.1: {}
      attr:
        algorithm: mul
    aten::linear_3539:
      type: InnerProduct
      input:
        ret1244.1: {}
        '728': {}
        aten::linear_3539_bias: {}
      output:
        ret1247.1: {}
    aten::mul_4972:
      type: Mul
      input:
        ret1243.1: {}
        ret1247.1: {}
      output:
        ret1248.1: {}
      attr:
        algorithm: mul
    aten::mul_567:
      type: Mul
      input:
        ret1248.1: {}
        '729': {}
      output:
        ret1249.1: {}
      attr:
        algorithm: mul
    aten::linear_3540:
      type: InnerProduct
      input:
        ret1249.1: {}
        '730': {}
        aten::linear_3540_bias: {}
      output:
        ret1252.1: {}
    aten::add_3171:
      type: Add
      input:
        ret1232.1: {}
        ret1252.1: {}
      output:
        ret1253.1: {}
    aten::pow_2733:
      type: Pow
      input:
        ret1253.1: {}
        aten::pow_2733_other: {}
      output:
        ret1254.1: {}
    aten::mean_1003:
      type: ReduceMean
      input:
        ret1254.1: {}
      output:
        ret1255.1: {}
    aten::add_38:
      type: Add
      input:
        ret1255.1: {}
        '485': {}
      output:
        ret1256.1: {}
    aten::rsqrt_4976:
      type: Rsqrt
      input:
        ret1256.1: {}
      output:
        ret1257.1: {}
    aten::mul_4975:
      type: Mul
      input:
        ret1253.1: {}
        ret1257.1: {}
      output:
        ret1258.1: {}
      attr:
        algorithm: mul
    aten::mul_569:
      type: Mul
      input:
        '731': {}
        ret1258.1: {}
      output:
        ret1259.1: {}
      attr:
        algorithm: mul
    aten::size_3404:
      type: Shape
      input:
        ret1259.1: {}
      output:
        '4757': {}
      attr:
        start: 0
        end: 0
    aten::size_3172:
      type: Shape
      input:
        ret1259.1: {}
      output:
        '4759': {}
      attr:
        start: 1
        end: 1
    aten::mul_570:
      type: Mul
      input:
        ret1259.1: {}
        '732': {}
      output:
        ret1262.1: {}
      attr:
        algorithm: mul
    aten::linear_3541:
      type: InnerProduct
      input:
        ret1262.1: {}
        '733': {}
        aten::linear_3541_bias: {}
      output:
        ret1265.1: {}
    aten::view_4978:
      type: View
      input:
        ret1265.1: {}
        '4757': {}
        '4759': {}
      output:
        ret1266.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2734:
      type: Reorder
      input:
        ret1266.1: {}
      output:
        ret1267.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_572:
      type: Mul
      input:
        ret1259.1: {}
        '734': {}
      output:
        ret1268.1: {}
      attr:
        algorithm: mul
    aten::linear_3542:
      type: InnerProduct
      input:
        ret1268.1: {}
        '735': {}
        aten::linear_3542_bias: {}
      output:
        ret1271.1: {}
    aten::view_4979:
      type: View
      input:
        ret1271.1: {}
        '4757': {}
        '4759': {}
      output:
        ret1272.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2735:
      type: Reorder
      input:
        ret1272.1: {}
      output:
        ret1273.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_574:
      type: Mul
      input:
        ret1259.1: {}
        '736': {}
      output:
        ret1274.1: {}
      attr:
        algorithm: mul
    aten::linear_3543:
      type: InnerProduct
      input:
        ret1274.1: {}
        '737': {}
        aten::linear_3543_bias: {}
      output:
        ret1277.1: {}
    aten::view_4980:
      type: View
      input:
        ret1277.1: {}
        '4757': {}
        '4759': {}
      output:
        ret1278.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2736:
      type: Reorder
      input:
        ret1278.1: {}
      output:
        ret1279.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2737:
      type: Shape
      input:
        ret1273.1: {}
      output:
        '4805': {}
      attr:
        start: 2
        end: 2
    aten::size_2738:
      type: Shape
      input:
        x29.1: {}
      output:
        '4808': {}
      attr:
        start: 2
        end: 2
    aten::add_3362:
      type: Add
      input:
        '4805': {}
        '4808': {}
      output:
        seq_len14.1: {}
    aten::slice_111:
      type: Slice
      input:
        '493': {}
        seq_len14.1: {}
      output:
        '4815': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_151:
      type: Slice
      input:
        '494': {}
        seq_len14.1: {}
      output:
        '4817': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2739:
      type: Shape
      input:
        ret1267.1: {}
      output:
        '4819': {}
      attr:
        start: 2
        end: 2
    aten::add_3173:
      type: Add
      input:
        '4819': {}
        '4808': {}
      output:
        '4824': {}
    aten::slice_2740:
      type: Slice
      input:
        '4815': {}
        '4808': {}
        '4824': {}
      output:
        '4826': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2180:
      type: Slice
      input:
        '4826': {}
      output:
        '4827': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2741:
      type: Slice
      input:
        '4817': {}
        '4808': {}
        '4824': {}
      output:
        '4828': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2181:
      type: Slice
      input:
        '4828': {}
      output:
        '4829': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_4981:
      type: Mul
      input:
        ret1267.1: {}
        '4827': {}
      output:
        ret1283.1: {}
      attr:
        algorithm: mul
    aten::size_2182:
      type: Shape
      input:
        ret1267.1: {}
      output:
        '4833': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_206:
      type: Div
      input:
        '4833': {}
        '495': {}
      output:
        ret1285.1: {}
      attr:
        algorithm: div
    aten::slice_2183:
      type: Slice
      input:
        ret1267.1: {}
        ret1285.1: {}
      output:
        ret1286.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2184:
      type: Slice
      input:
        ret1267.1: {}
        ret1285.1: {}
      output:
        ret1287.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4988:
      type: Neg
      input:
        ret1287.1: {}
        aten::neg_4988_mul_val: {}
      output:
        ret1288.1: {}
      attr:
        algorithm: mul
    aten::cat_2483:
      type: Concat
      input:
        ret1288.1: {}
        ret1286.1: {}
      output:
        ret1289.1: {}
      attr:
        axis: -1
    aten::mul_4985:
      type: Mul
      input:
        ret1289.1: {}
        '4829': {}
      output:
        ret1290.1: {}
      attr:
        algorithm: mul
    aten::add_3174:
      type: Add
      input:
        ret1283.1: {}
        ret1290.1: {}
      output:
        args59.1: {}
    aten::mul_4983:
      type: Mul
      input:
        ret1273.1: {}
        '4827': {}
      output:
        ret1291.1: {}
      attr:
        algorithm: mul
    aten::size_2185:
      type: Shape
      input:
        ret1273.1: {}
      output:
        '4855': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_207:
      type: Div
      input:
        '4855': {}
        '495': {}
      output:
        ret1293.1: {}
      attr:
        algorithm: div
    aten::slice_2186:
      type: Slice
      input:
        ret1273.1: {}
        ret1293.1: {}
      output:
        ret1294.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2187:
      type: Slice
      input:
        ret1273.1: {}
        ret1293.1: {}
      output:
        ret1295.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_4990:
      type: Neg
      input:
        ret1295.1: {}
        aten::neg_4990_mul_val: {}
      output:
        ret1296.1: {}
      attr:
        algorithm: mul
    aten::cat_2484:
      type: Concat
      input:
        ret1296.1: {}
        ret1294.1: {}
      output:
        ret1297.1: {}
      attr:
        axis: -1
    aten::mul_4986:
      type: Mul
      input:
        ret1297.1: {}
        '4829': {}
      output:
        ret1298.1: {}
      attr:
        algorithm: mul
    aten::add_3175:
      type: Add
      input:
        ret1291.1: {}
        ret1298.1: {}
      output:
        '4873': {}
    aten::cat_2742:
      type: Concat
      input:
        x29.1: {}
        '4873': {}
      output:
        ret1299.1: {}
      attr:
        axis: 2
    aten::cat_2743:
      type: Concat
      input:
        x30.1: {}
        ret1279.1: {}
      output:
        ret1300.1: {}
      attr:
        axis: 2
    aten::transpose_2188:
      type: Reorder
      input:
        ret1299.1: {}
      output:
        ret1301.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_4993:
      type: Matmul
      input:
        args59.1: {}
        ret1301.1: {}
      output:
        ret1304.1: {}
    aten::div_271:
      type: Div
      input:
        ret1304.1: {}
        '496': {}
      output:
        ret1305.1: {}
      attr:
        algorithm: div
    aten::add_3176:
      type: Add
      input:
        ret1305.1: {}
        attention_mask0.1: {}
      output:
        attn_weights30.1: {}
    aten::max_311:
      type: Max
      input:
        attn_weights30.1: {}
        '497': {}
      output:
        input30.1: {}
    aten::softmax_2424:
      type: Softmax
      input:
        input30.1: {}
      output:
        '4895': {}
      attr:
        axis: -1
    aten::matmul_4996:
      type: Matmul
      input:
        '4895': {}
        ret1300.1: {}
      output:
        ret1308.1: {}
    aten::transpose_2744:
      type: Reorder
      input:
        ret1308.1: {}
      output:
        ret1309.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_4998:
      type: Reshape
      input:
        ret1309.1: {}
        '4757': {}
        '4759': {}
      output:
        ret1310.1: {}
    aten::mul_576:
      type: Mul
      input:
        ret1310.1: {}
        '738': {}
      output:
        ret1311.1: {}
      attr:
        algorithm: mul
    aten::linear_3544:
      type: InnerProduct
      input:
        ret1311.1: {}
        '739': {}
        aten::linear_3544_bias: {}
      output:
        ret1314.1: {}
    aten::add_3177:
      type: Add
      input:
        ret1253.1: {}
        ret1314.1: {}
      output:
        ret1315.1: {}
    aten::pow_2745:
      type: Pow
      input:
        ret1315.1: {}
        aten::pow_2745_other: {}
      output:
        ret1316.1: {}
    aten::mean_1004:
      type: ReduceMean
      input:
        ret1316.1: {}
      output:
        ret1317.1: {}
    aten::add_39:
      type: Add
      input:
        ret1317.1: {}
        '485': {}
      output:
        ret1318.1: {}
    aten::rsqrt_5001:
      type: Rsqrt
      input:
        ret1318.1: {}
      output:
        ret1319.1: {}
    aten::mul_5000:
      type: Mul
      input:
        ret1315.1: {}
        ret1319.1: {}
      output:
        ret1320.1: {}
      attr:
        algorithm: mul
    aten::mul_578:
      type: Mul
      input:
        '740': {}
        ret1320.1: {}
      output:
        ret1321.1: {}
      attr:
        algorithm: mul
    aten::mul_579:
      type: Mul
      input:
        ret1321.1: {}
        '741': {}
      output:
        ret1322.1: {}
      attr:
        algorithm: mul
    aten::linear_3545:
      type: InnerProduct
      input:
        ret1322.1: {}
        '742': {}
        aten::linear_3545_bias: {}
      output:
        ret1325.1: {}
    aten::silu_5003:
      type: Swish
      input:
        ret1325.1: {}
      output:
        ret1326.1: {}
    aten::mul_581:
      type: Mul
      input:
        ret1321.1: {}
        '743': {}
      output:
        ret1327.1: {}
      attr:
        algorithm: mul
    aten::linear_3546:
      type: InnerProduct
      input:
        ret1327.1: {}
        '744': {}
        aten::linear_3546_bias: {}
      output:
        ret1330.1: {}
    aten::mul_5004:
      type: Mul
      input:
        ret1326.1: {}
        ret1330.1: {}
      output:
        ret1331.1: {}
      attr:
        algorithm: mul
    aten::mul_583:
      type: Mul
      input:
        ret1331.1: {}
        '745': {}
      output:
        ret1332.1: {}
      attr:
        algorithm: mul
    aten::linear_3547:
      type: InnerProduct
      input:
        ret1332.1: {}
        '746': {}
        aten::linear_3547_bias: {}
      output:
        ret1335.1: {}
    aten::add_3178:
      type: Add
      input:
        ret1315.1: {}
        ret1335.1: {}
      output:
        ret1336.1: {}
    aten::pow_2746:
      type: Pow
      input:
        ret1336.1: {}
        aten::pow_2746_other: {}
      output:
        ret1337.1: {}
    aten::mean_1005:
      type: ReduceMean
      input:
        ret1337.1: {}
      output:
        ret1338.1: {}
    aten::add_40:
      type: Add
      input:
        ret1338.1: {}
        '485': {}
      output:
        ret1339.1: {}
    aten::rsqrt_5008:
      type: Rsqrt
      input:
        ret1339.1: {}
      output:
        ret1340.1: {}
    aten::mul_5007:
      type: Mul
      input:
        ret1336.1: {}
        ret1340.1: {}
      output:
        ret1341.1: {}
      attr:
        algorithm: mul
    aten::mul_585:
      type: Mul
      input:
        '747': {}
        ret1341.1: {}
      output:
        ret1342.1: {}
      attr:
        algorithm: mul
    aten::size_3405:
      type: Shape
      input:
        ret1342.1: {}
      output:
        '4980': {}
      attr:
        start: 0
        end: 0
    aten::size_3179:
      type: Shape
      input:
        ret1342.1: {}
      output:
        '4982': {}
      attr:
        start: 1
        end: 1
    aten::mul_586:
      type: Mul
      input:
        ret1342.1: {}
        '748': {}
      output:
        ret1345.1: {}
      attr:
        algorithm: mul
    aten::linear_3548:
      type: InnerProduct
      input:
        ret1345.1: {}
        '749': {}
        aten::linear_3548_bias: {}
      output:
        ret1348.1: {}
    aten::view_5010:
      type: View
      input:
        ret1348.1: {}
        '4980': {}
        '4982': {}
      output:
        ret1349.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2747:
      type: Reorder
      input:
        ret1349.1: {}
      output:
        ret1350.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_588:
      type: Mul
      input:
        ret1342.1: {}
        '750': {}
      output:
        ret1351.1: {}
      attr:
        algorithm: mul
    aten::linear_3549:
      type: InnerProduct
      input:
        ret1351.1: {}
        '751': {}
        aten::linear_3549_bias: {}
      output:
        ret1354.1: {}
    aten::view_5011:
      type: View
      input:
        ret1354.1: {}
        '4980': {}
        '4982': {}
      output:
        ret1355.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2748:
      type: Reorder
      input:
        ret1355.1: {}
      output:
        ret1356.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_590:
      type: Mul
      input:
        ret1342.1: {}
        '752': {}
      output:
        ret1357.1: {}
      attr:
        algorithm: mul
    aten::linear_3550:
      type: InnerProduct
      input:
        ret1357.1: {}
        '753': {}
        aten::linear_3550_bias: {}
      output:
        ret1360.1: {}
    aten::view_5012:
      type: View
      input:
        ret1360.1: {}
        '4980': {}
        '4982': {}
      output:
        ret1361.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2749:
      type: Reorder
      input:
        ret1361.1: {}
      output:
        ret1362.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2750:
      type: Shape
      input:
        ret1356.1: {}
      output:
        '5028': {}
      attr:
        start: 2
        end: 2
    aten::size_2751:
      type: Shape
      input:
        x31.1: {}
      output:
        '5031': {}
      attr:
        start: 2
        end: 2
    aten::add_3363:
      type: Add
      input:
        '5028': {}
        '5031': {}
      output:
        seq_len15.1: {}
    aten::slice_112:
      type: Slice
      input:
        '493': {}
        seq_len15.1: {}
      output:
        '5038': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_152:
      type: Slice
      input:
        '494': {}
        seq_len15.1: {}
      output:
        '5040': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2752:
      type: Shape
      input:
        ret1350.1: {}
      output:
        '5042': {}
      attr:
        start: 2
        end: 2
    aten::add_3180:
      type: Add
      input:
        '5042': {}
        '5031': {}
      output:
        '5047': {}
    aten::slice_2753:
      type: Slice
      input:
        '5038': {}
        '5031': {}
        '5047': {}
      output:
        '5049': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2189:
      type: Slice
      input:
        '5049': {}
      output:
        '5050': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2754:
      type: Slice
      input:
        '5040': {}
        '5031': {}
        '5047': {}
      output:
        '5051': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2190:
      type: Slice
      input:
        '5051': {}
      output:
        '5052': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5013:
      type: Mul
      input:
        ret1350.1: {}
        '5050': {}
      output:
        ret1366.1: {}
      attr:
        algorithm: mul
    aten::size_2191:
      type: Shape
      input:
        ret1350.1: {}
      output:
        '5056': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_208:
      type: Div
      input:
        '5056': {}
        '495': {}
      output:
        ret1368.1: {}
      attr:
        algorithm: div
    aten::slice_2192:
      type: Slice
      input:
        ret1350.1: {}
        ret1368.1: {}
      output:
        ret1369.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2193:
      type: Slice
      input:
        ret1350.1: {}
        ret1368.1: {}
      output:
        ret1370.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5020:
      type: Neg
      input:
        ret1370.1: {}
        aten::neg_5020_mul_val: {}
      output:
        ret1371.1: {}
      attr:
        algorithm: mul
    aten::cat_2485:
      type: Concat
      input:
        ret1371.1: {}
        ret1369.1: {}
      output:
        ret1372.1: {}
      attr:
        axis: -1
    aten::mul_5017:
      type: Mul
      input:
        ret1372.1: {}
        '5052': {}
      output:
        ret1373.1: {}
      attr:
        algorithm: mul
    aten::add_3181:
      type: Add
      input:
        ret1366.1: {}
        ret1373.1: {}
      output:
        args63.1: {}
    aten::mul_5015:
      type: Mul
      input:
        ret1356.1: {}
        '5050': {}
      output:
        ret1374.1: {}
      attr:
        algorithm: mul
    aten::size_2194:
      type: Shape
      input:
        ret1356.1: {}
      output:
        '5078': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_209:
      type: Div
      input:
        '5078': {}
        '495': {}
      output:
        ret1376.1: {}
      attr:
        algorithm: div
    aten::slice_2195:
      type: Slice
      input:
        ret1356.1: {}
        ret1376.1: {}
      output:
        ret1377.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2196:
      type: Slice
      input:
        ret1356.1: {}
        ret1376.1: {}
      output:
        ret1378.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5022:
      type: Neg
      input:
        ret1378.1: {}
        aten::neg_5022_mul_val: {}
      output:
        ret1379.1: {}
      attr:
        algorithm: mul
    aten::cat_2486:
      type: Concat
      input:
        ret1379.1: {}
        ret1377.1: {}
      output:
        ret1380.1: {}
      attr:
        axis: -1
    aten::mul_5018:
      type: Mul
      input:
        ret1380.1: {}
        '5052': {}
      output:
        ret1381.1: {}
      attr:
        algorithm: mul
    aten::add_3182:
      type: Add
      input:
        ret1374.1: {}
        ret1381.1: {}
      output:
        '5096': {}
    aten::cat_2755:
      type: Concat
      input:
        x31.1: {}
        '5096': {}
      output:
        ret1382.1: {}
      attr:
        axis: 2
    aten::cat_2756:
      type: Concat
      input:
        x32.1: {}
        ret1362.1: {}
      output:
        ret1383.1: {}
      attr:
        axis: 2
    aten::transpose_2197:
      type: Reorder
      input:
        ret1382.1: {}
      output:
        ret1384.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5025:
      type: Matmul
      input:
        args63.1: {}
        ret1384.1: {}
      output:
        ret1387.1: {}
    aten::div_272:
      type: Div
      input:
        ret1387.1: {}
        '496': {}
      output:
        ret1388.1: {}
      attr:
        algorithm: div
    aten::add_3183:
      type: Add
      input:
        ret1388.1: {}
        attention_mask0.1: {}
      output:
        attn_weights32.1: {}
    aten::max_312:
      type: Max
      input:
        attn_weights32.1: {}
        '497': {}
      output:
        input32.1: {}
    aten::softmax_2425:
      type: Softmax
      input:
        input32.1: {}
      output:
        '5118': {}
      attr:
        axis: -1
    aten::matmul_5028:
      type: Matmul
      input:
        '5118': {}
        ret1383.1: {}
      output:
        ret1391.1: {}
    aten::transpose_2757:
      type: Reorder
      input:
        ret1391.1: {}
      output:
        ret1392.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5030:
      type: Reshape
      input:
        ret1392.1: {}
        '4980': {}
        '4982': {}
      output:
        ret1393.1: {}
    aten::mul_592:
      type: Mul
      input:
        ret1393.1: {}
        '754': {}
      output:
        ret1394.1: {}
      attr:
        algorithm: mul
    aten::linear_3551:
      type: InnerProduct
      input:
        ret1394.1: {}
        '755': {}
        aten::linear_3551_bias: {}
      output:
        ret1397.1: {}
    aten::add_3184:
      type: Add
      input:
        ret1336.1: {}
        ret1397.1: {}
      output:
        ret1398.1: {}
    aten::pow_2758:
      type: Pow
      input:
        ret1398.1: {}
        aten::pow_2758_other: {}
      output:
        ret1399.1: {}
    aten::mean_1006:
      type: ReduceMean
      input:
        ret1399.1: {}
      output:
        ret1400.1: {}
    aten::add_41:
      type: Add
      input:
        ret1400.1: {}
        '485': {}
      output:
        ret1401.1: {}
    aten::rsqrt_5033:
      type: Rsqrt
      input:
        ret1401.1: {}
      output:
        ret1402.1: {}
    aten::mul_5032:
      type: Mul
      input:
        ret1398.1: {}
        ret1402.1: {}
      output:
        ret1403.1: {}
      attr:
        algorithm: mul
    aten::mul_594:
      type: Mul
      input:
        '756': {}
        ret1403.1: {}
      output:
        ret1404.1: {}
      attr:
        algorithm: mul
    aten::mul_595:
      type: Mul
      input:
        ret1404.1: {}
        '757': {}
      output:
        ret1405.1: {}
      attr:
        algorithm: mul
    aten::linear_3552:
      type: InnerProduct
      input:
        ret1405.1: {}
        '758': {}
        aten::linear_3552_bias: {}
      output:
        ret1408.1: {}
    aten::silu_5035:
      type: Swish
      input:
        ret1408.1: {}
      output:
        ret1409.1: {}
    aten::mul_597:
      type: Mul
      input:
        ret1404.1: {}
        '759': {}
      output:
        ret1410.1: {}
      attr:
        algorithm: mul
    aten::linear_3553:
      type: InnerProduct
      input:
        ret1410.1: {}
        '760': {}
        aten::linear_3553_bias: {}
      output:
        ret1413.1: {}
    aten::mul_5036:
      type: Mul
      input:
        ret1409.1: {}
        ret1413.1: {}
      output:
        ret1414.1: {}
      attr:
        algorithm: mul
    aten::mul_599:
      type: Mul
      input:
        ret1414.1: {}
        '761': {}
      output:
        ret1415.1: {}
      attr:
        algorithm: mul
    aten::linear_3554:
      type: InnerProduct
      input:
        ret1415.1: {}
        '762': {}
        aten::linear_3554_bias: {}
      output:
        ret1418.1: {}
    aten::add_3185:
      type: Add
      input:
        ret1398.1: {}
        ret1418.1: {}
      output:
        ret1419.1: {}
    aten::pow_2759:
      type: Pow
      input:
        ret1419.1: {}
        aten::pow_2759_other: {}
      output:
        ret1420.1: {}
    aten::mean_1007:
      type: ReduceMean
      input:
        ret1420.1: {}
      output:
        ret1421.1: {}
    aten::add_42:
      type: Add
      input:
        ret1421.1: {}
        '485': {}
      output:
        ret1422.1: {}
    aten::rsqrt_5040:
      type: Rsqrt
      input:
        ret1422.1: {}
      output:
        ret1423.1: {}
    aten::mul_5039:
      type: Mul
      input:
        ret1419.1: {}
        ret1423.1: {}
      output:
        ret1424.1: {}
      attr:
        algorithm: mul
    aten::mul_601:
      type: Mul
      input:
        '763': {}
        ret1424.1: {}
      output:
        ret1425.1: {}
      attr:
        algorithm: mul
    aten::size_3406:
      type: Shape
      input:
        ret1425.1: {}
      output:
        '5203': {}
      attr:
        start: 0
        end: 0
    aten::size_3186:
      type: Shape
      input:
        ret1425.1: {}
      output:
        '5205': {}
      attr:
        start: 1
        end: 1
    aten::mul_602:
      type: Mul
      input:
        ret1425.1: {}
        '764': {}
      output:
        ret1428.1: {}
      attr:
        algorithm: mul
    aten::linear_3555:
      type: InnerProduct
      input:
        ret1428.1: {}
        '765': {}
        aten::linear_3555_bias: {}
      output:
        ret1431.1: {}
    aten::view_5042:
      type: View
      input:
        ret1431.1: {}
        '5203': {}
        '5205': {}
      output:
        ret1432.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2760:
      type: Reorder
      input:
        ret1432.1: {}
      output:
        ret1433.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_604:
      type: Mul
      input:
        ret1425.1: {}
        '766': {}
      output:
        ret1434.1: {}
      attr:
        algorithm: mul
    aten::linear_3556:
      type: InnerProduct
      input:
        ret1434.1: {}
        '767': {}
        aten::linear_3556_bias: {}
      output:
        ret1437.1: {}
    aten::view_5043:
      type: View
      input:
        ret1437.1: {}
        '5203': {}
        '5205': {}
      output:
        ret1438.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2761:
      type: Reorder
      input:
        ret1438.1: {}
      output:
        ret1439.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_606:
      type: Mul
      input:
        ret1425.1: {}
        '768': {}
      output:
        ret1440.1: {}
      attr:
        algorithm: mul
    aten::linear_3557:
      type: InnerProduct
      input:
        ret1440.1: {}
        '769': {}
        aten::linear_3557_bias: {}
      output:
        ret1443.1: {}
    aten::view_5044:
      type: View
      input:
        ret1443.1: {}
        '5203': {}
        '5205': {}
      output:
        ret1444.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2762:
      type: Reorder
      input:
        ret1444.1: {}
      output:
        ret1445.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2763:
      type: Shape
      input:
        ret1439.1: {}
      output:
        '5251': {}
      attr:
        start: 2
        end: 2
    aten::size_2764:
      type: Shape
      input:
        x33.1: {}
      output:
        '5254': {}
      attr:
        start: 2
        end: 2
    aten::add_3364:
      type: Add
      input:
        '5251': {}
        '5254': {}
      output:
        seq_len16.1: {}
    aten::slice_113:
      type: Slice
      input:
        '493': {}
        seq_len16.1: {}
      output:
        '5261': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_153:
      type: Slice
      input:
        '494': {}
        seq_len16.1: {}
      output:
        '5263': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2765:
      type: Shape
      input:
        ret1433.1: {}
      output:
        '5265': {}
      attr:
        start: 2
        end: 2
    aten::add_3187:
      type: Add
      input:
        '5265': {}
        '5254': {}
      output:
        '5270': {}
    aten::slice_2766:
      type: Slice
      input:
        '5261': {}
        '5254': {}
        '5270': {}
      output:
        '5272': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2198:
      type: Slice
      input:
        '5272': {}
      output:
        '5273': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2767:
      type: Slice
      input:
        '5263': {}
        '5254': {}
        '5270': {}
      output:
        '5274': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2199:
      type: Slice
      input:
        '5274': {}
      output:
        '5275': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5045:
      type: Mul
      input:
        ret1433.1: {}
        '5273': {}
      output:
        ret1449.1: {}
      attr:
        algorithm: mul
    aten::size_2200:
      type: Shape
      input:
        ret1433.1: {}
      output:
        '5279': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_210:
      type: Div
      input:
        '5279': {}
        '495': {}
      output:
        ret1451.1: {}
      attr:
        algorithm: div
    aten::slice_2201:
      type: Slice
      input:
        ret1433.1: {}
        ret1451.1: {}
      output:
        ret1452.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2202:
      type: Slice
      input:
        ret1433.1: {}
        ret1451.1: {}
      output:
        ret1453.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5052:
      type: Neg
      input:
        ret1453.1: {}
        aten::neg_5052_mul_val: {}
      output:
        ret1454.1: {}
      attr:
        algorithm: mul
    aten::cat_2487:
      type: Concat
      input:
        ret1454.1: {}
        ret1452.1: {}
      output:
        ret1455.1: {}
      attr:
        axis: -1
    aten::mul_5049:
      type: Mul
      input:
        ret1455.1: {}
        '5275': {}
      output:
        ret1456.1: {}
      attr:
        algorithm: mul
    aten::add_3188:
      type: Add
      input:
        ret1449.1: {}
        ret1456.1: {}
      output:
        args67.1: {}
    aten::mul_5047:
      type: Mul
      input:
        ret1439.1: {}
        '5273': {}
      output:
        ret1457.1: {}
      attr:
        algorithm: mul
    aten::size_2203:
      type: Shape
      input:
        ret1439.1: {}
      output:
        '5301': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_211:
      type: Div
      input:
        '5301': {}
        '495': {}
      output:
        ret1459.1: {}
      attr:
        algorithm: div
    aten::slice_2204:
      type: Slice
      input:
        ret1439.1: {}
        ret1459.1: {}
      output:
        ret1460.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2205:
      type: Slice
      input:
        ret1439.1: {}
        ret1459.1: {}
      output:
        ret1461.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5054:
      type: Neg
      input:
        ret1461.1: {}
        aten::neg_5054_mul_val: {}
      output:
        ret1462.1: {}
      attr:
        algorithm: mul
    aten::cat_2488:
      type: Concat
      input:
        ret1462.1: {}
        ret1460.1: {}
      output:
        ret1463.1: {}
      attr:
        axis: -1
    aten::mul_5050:
      type: Mul
      input:
        ret1463.1: {}
        '5275': {}
      output:
        ret1464.1: {}
      attr:
        algorithm: mul
    aten::add_3189:
      type: Add
      input:
        ret1457.1: {}
        ret1464.1: {}
      output:
        '5319': {}
    aten::cat_2768:
      type: Concat
      input:
        x33.1: {}
        '5319': {}
      output:
        ret1465.1: {}
      attr:
        axis: 2
    aten::cat_2769:
      type: Concat
      input:
        x34.1: {}
        ret1445.1: {}
      output:
        ret1466.1: {}
      attr:
        axis: 2
    aten::transpose_2206:
      type: Reorder
      input:
        ret1465.1: {}
      output:
        ret1467.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5057:
      type: Matmul
      input:
        args67.1: {}
        ret1467.1: {}
      output:
        ret1470.1: {}
    aten::div_273:
      type: Div
      input:
        ret1470.1: {}
        '496': {}
      output:
        ret1471.1: {}
      attr:
        algorithm: div
    aten::add_3190:
      type: Add
      input:
        ret1471.1: {}
        attention_mask0.1: {}
      output:
        attn_weights34.1: {}
    aten::max_313:
      type: Max
      input:
        attn_weights34.1: {}
        '497': {}
      output:
        input34.1: {}
    aten::softmax_2426:
      type: Softmax
      input:
        input34.1: {}
      output:
        '5341': {}
      attr:
        axis: -1
    aten::matmul_5060:
      type: Matmul
      input:
        '5341': {}
        ret1466.1: {}
      output:
        ret1474.1: {}
    aten::transpose_2770:
      type: Reorder
      input:
        ret1474.1: {}
      output:
        ret1475.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5062:
      type: Reshape
      input:
        ret1475.1: {}
        '5203': {}
        '5205': {}
      output:
        ret1476.1: {}
    aten::mul_608:
      type: Mul
      input:
        ret1476.1: {}
        '770': {}
      output:
        ret1477.1: {}
      attr:
        algorithm: mul
    aten::linear_3558:
      type: InnerProduct
      input:
        ret1477.1: {}
        '771': {}
        aten::linear_3558_bias: {}
      output:
        ret1480.1: {}
    aten::add_3191:
      type: Add
      input:
        ret1419.1: {}
        ret1480.1: {}
      output:
        ret1481.1: {}
    aten::pow_2771:
      type: Pow
      input:
        ret1481.1: {}
        aten::pow_2771_other: {}
      output:
        ret1482.1: {}
    aten::mean_1008:
      type: ReduceMean
      input:
        ret1482.1: {}
      output:
        ret1483.1: {}
    aten::add_43:
      type: Add
      input:
        ret1483.1: {}
        '485': {}
      output:
        ret1484.1: {}
    aten::rsqrt_5065:
      type: Rsqrt
      input:
        ret1484.1: {}
      output:
        ret1485.1: {}
    aten::mul_5064:
      type: Mul
      input:
        ret1481.1: {}
        ret1485.1: {}
      output:
        ret1486.1: {}
      attr:
        algorithm: mul
    aten::mul_610:
      type: Mul
      input:
        '772': {}
        ret1486.1: {}
      output:
        ret1487.1: {}
      attr:
        algorithm: mul
    aten::mul_611:
      type: Mul
      input:
        ret1487.1: {}
        '773': {}
      output:
        ret1488.1: {}
      attr:
        algorithm: mul
    aten::linear_3559:
      type: InnerProduct
      input:
        ret1488.1: {}
        '774': {}
        aten::linear_3559_bias: {}
      output:
        ret1491.1: {}
    aten::silu_5067:
      type: Swish
      input:
        ret1491.1: {}
      output:
        ret1492.1: {}
    aten::mul_613:
      type: Mul
      input:
        ret1487.1: {}
        '775': {}
      output:
        ret1493.1: {}
      attr:
        algorithm: mul
    aten::linear_3560:
      type: InnerProduct
      input:
        ret1493.1: {}
        '776': {}
        aten::linear_3560_bias: {}
      output:
        ret1496.1: {}
    aten::mul_5068:
      type: Mul
      input:
        ret1492.1: {}
        ret1496.1: {}
      output:
        ret1497.1: {}
      attr:
        algorithm: mul
    aten::mul_615:
      type: Mul
      input:
        ret1497.1: {}
        '777': {}
      output:
        ret1498.1: {}
      attr:
        algorithm: mul
    aten::linear_3561:
      type: InnerProduct
      input:
        ret1498.1: {}
        '778': {}
        aten::linear_3561_bias: {}
      output:
        ret1501.1: {}
    aten::add_3192:
      type: Add
      input:
        ret1481.1: {}
        ret1501.1: {}
      output:
        ret1502.1: {}
    aten::pow_2772:
      type: Pow
      input:
        ret1502.1: {}
        aten::pow_2772_other: {}
      output:
        ret1503.1: {}
    aten::mean_1009:
      type: ReduceMean
      input:
        ret1503.1: {}
      output:
        ret1504.1: {}
    aten::add_44:
      type: Add
      input:
        ret1504.1: {}
        '485': {}
      output:
        ret1505.1: {}
    aten::rsqrt_5072:
      type: Rsqrt
      input:
        ret1505.1: {}
      output:
        ret1506.1: {}
    aten::mul_5071:
      type: Mul
      input:
        ret1502.1: {}
        ret1506.1: {}
      output:
        ret1507.1: {}
      attr:
        algorithm: mul
    aten::mul_617:
      type: Mul
      input:
        '779': {}
        ret1507.1: {}
      output:
        ret1508.1: {}
      attr:
        algorithm: mul
    aten::size_3407:
      type: Shape
      input:
        ret1508.1: {}
      output:
        '5426': {}
      attr:
        start: 0
        end: 0
    aten::size_3193:
      type: Shape
      input:
        ret1508.1: {}
      output:
        '5428': {}
      attr:
        start: 1
        end: 1
    aten::mul_618:
      type: Mul
      input:
        ret1508.1: {}
        '780': {}
      output:
        ret1511.1: {}
      attr:
        algorithm: mul
    aten::linear_3562:
      type: InnerProduct
      input:
        ret1511.1: {}
        '781': {}
        aten::linear_3562_bias: {}
      output:
        ret1514.1: {}
    aten::view_5074:
      type: View
      input:
        ret1514.1: {}
        '5426': {}
        '5428': {}
      output:
        ret1515.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2773:
      type: Reorder
      input:
        ret1515.1: {}
      output:
        ret1516.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_620:
      type: Mul
      input:
        ret1508.1: {}
        '782': {}
      output:
        ret1517.1: {}
      attr:
        algorithm: mul
    aten::linear_3563:
      type: InnerProduct
      input:
        ret1517.1: {}
        '783': {}
        aten::linear_3563_bias: {}
      output:
        ret1520.1: {}
    aten::view_5075:
      type: View
      input:
        ret1520.1: {}
        '5426': {}
        '5428': {}
      output:
        ret1521.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2774:
      type: Reorder
      input:
        ret1521.1: {}
      output:
        ret1522.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_622:
      type: Mul
      input:
        ret1508.1: {}
        '784': {}
      output:
        ret1523.1: {}
      attr:
        algorithm: mul
    aten::linear_3564:
      type: InnerProduct
      input:
        ret1523.1: {}
        '785': {}
        aten::linear_3564_bias: {}
      output:
        ret1526.1: {}
    aten::view_5076:
      type: View
      input:
        ret1526.1: {}
        '5426': {}
        '5428': {}
      output:
        ret1527.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2775:
      type: Reorder
      input:
        ret1527.1: {}
      output:
        ret1528.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2776:
      type: Shape
      input:
        ret1522.1: {}
      output:
        '5474': {}
      attr:
        start: 2
        end: 2
    aten::size_2777:
      type: Shape
      input:
        x35.1: {}
      output:
        '5477': {}
      attr:
        start: 2
        end: 2
    aten::add_3365:
      type: Add
      input:
        '5474': {}
        '5477': {}
      output:
        seq_len17.1: {}
    aten::slice_114:
      type: Slice
      input:
        '493': {}
        seq_len17.1: {}
      output:
        '5484': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_154:
      type: Slice
      input:
        '494': {}
        seq_len17.1: {}
      output:
        '5486': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2778:
      type: Shape
      input:
        ret1516.1: {}
      output:
        '5488': {}
      attr:
        start: 2
        end: 2
    aten::add_3194:
      type: Add
      input:
        '5488': {}
        '5477': {}
      output:
        '5493': {}
    aten::slice_2779:
      type: Slice
      input:
        '5484': {}
        '5477': {}
        '5493': {}
      output:
        '5495': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2207:
      type: Slice
      input:
        '5495': {}
      output:
        '5496': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2780:
      type: Slice
      input:
        '5486': {}
        '5477': {}
        '5493': {}
      output:
        '5497': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2208:
      type: Slice
      input:
        '5497': {}
      output:
        '5498': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5077:
      type: Mul
      input:
        ret1516.1: {}
        '5496': {}
      output:
        ret1532.1: {}
      attr:
        algorithm: mul
    aten::size_2209:
      type: Shape
      input:
        ret1516.1: {}
      output:
        '5502': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_212:
      type: Div
      input:
        '5502': {}
        '495': {}
      output:
        ret1534.1: {}
      attr:
        algorithm: div
    aten::slice_2210:
      type: Slice
      input:
        ret1516.1: {}
        ret1534.1: {}
      output:
        ret1535.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2211:
      type: Slice
      input:
        ret1516.1: {}
        ret1534.1: {}
      output:
        ret1536.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5084:
      type: Neg
      input:
        ret1536.1: {}
        aten::neg_5084_mul_val: {}
      output:
        ret1537.1: {}
      attr:
        algorithm: mul
    aten::cat_2489:
      type: Concat
      input:
        ret1537.1: {}
        ret1535.1: {}
      output:
        ret1538.1: {}
      attr:
        axis: -1
    aten::mul_5081:
      type: Mul
      input:
        ret1538.1: {}
        '5498': {}
      output:
        ret1539.1: {}
      attr:
        algorithm: mul
    aten::add_3195:
      type: Add
      input:
        ret1532.1: {}
        ret1539.1: {}
      output:
        args71.1: {}
    aten::mul_5079:
      type: Mul
      input:
        ret1522.1: {}
        '5496': {}
      output:
        ret1540.1: {}
      attr:
        algorithm: mul
    aten::size_2212:
      type: Shape
      input:
        ret1522.1: {}
      output:
        '5524': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_213:
      type: Div
      input:
        '5524': {}
        '495': {}
      output:
        ret1542.1: {}
      attr:
        algorithm: div
    aten::slice_2213:
      type: Slice
      input:
        ret1522.1: {}
        ret1542.1: {}
      output:
        ret1543.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2214:
      type: Slice
      input:
        ret1522.1: {}
        ret1542.1: {}
      output:
        ret1544.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5086:
      type: Neg
      input:
        ret1544.1: {}
        aten::neg_5086_mul_val: {}
      output:
        ret1545.1: {}
      attr:
        algorithm: mul
    aten::cat_2490:
      type: Concat
      input:
        ret1545.1: {}
        ret1543.1: {}
      output:
        ret1546.1: {}
      attr:
        axis: -1
    aten::mul_5082:
      type: Mul
      input:
        ret1546.1: {}
        '5498': {}
      output:
        ret1547.1: {}
      attr:
        algorithm: mul
    aten::add_3196:
      type: Add
      input:
        ret1540.1: {}
        ret1547.1: {}
      output:
        '5542': {}
    aten::cat_2781:
      type: Concat
      input:
        x35.1: {}
        '5542': {}
      output:
        ret1548.1: {}
      attr:
        axis: 2
    aten::cat_2782:
      type: Concat
      input:
        x36.1: {}
        ret1528.1: {}
      output:
        ret1549.1: {}
      attr:
        axis: 2
    aten::transpose_2215:
      type: Reorder
      input:
        ret1548.1: {}
      output:
        ret1550.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5089:
      type: Matmul
      input:
        args71.1: {}
        ret1550.1: {}
      output:
        ret1553.1: {}
    aten::div_274:
      type: Div
      input:
        ret1553.1: {}
        '496': {}
      output:
        ret1554.1: {}
      attr:
        algorithm: div
    aten::add_3197:
      type: Add
      input:
        ret1554.1: {}
        attention_mask0.1: {}
      output:
        attn_weights36.1: {}
    aten::max_314:
      type: Max
      input:
        attn_weights36.1: {}
        '497': {}
      output:
        input36.1: {}
    aten::softmax_2427:
      type: Softmax
      input:
        input36.1: {}
      output:
        '5564': {}
      attr:
        axis: -1
    aten::matmul_5092:
      type: Matmul
      input:
        '5564': {}
        ret1549.1: {}
      output:
        ret1557.1: {}
    aten::transpose_2783:
      type: Reorder
      input:
        ret1557.1: {}
      output:
        ret1558.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5094:
      type: Reshape
      input:
        ret1558.1: {}
        '5426': {}
        '5428': {}
      output:
        ret1559.1: {}
    aten::mul_624:
      type: Mul
      input:
        ret1559.1: {}
        '786': {}
      output:
        ret1560.1: {}
      attr:
        algorithm: mul
    aten::linear_3565:
      type: InnerProduct
      input:
        ret1560.1: {}
        '787': {}
        aten::linear_3565_bias: {}
      output:
        ret1563.1: {}
    aten::add_3198:
      type: Add
      input:
        ret1502.1: {}
        ret1563.1: {}
      output:
        ret1564.1: {}
    aten::pow_2784:
      type: Pow
      input:
        ret1564.1: {}
        aten::pow_2784_other: {}
      output:
        ret1565.1: {}
    aten::mean_1010:
      type: ReduceMean
      input:
        ret1565.1: {}
      output:
        ret1566.1: {}
    aten::add_45:
      type: Add
      input:
        ret1566.1: {}
        '485': {}
      output:
        ret1567.1: {}
    aten::rsqrt_5097:
      type: Rsqrt
      input:
        ret1567.1: {}
      output:
        ret1568.1: {}
    aten::mul_5096:
      type: Mul
      input:
        ret1564.1: {}
        ret1568.1: {}
      output:
        ret1569.1: {}
      attr:
        algorithm: mul
    aten::mul_626:
      type: Mul
      input:
        '788': {}
        ret1569.1: {}
      output:
        ret1570.1: {}
      attr:
        algorithm: mul
    aten::mul_627:
      type: Mul
      input:
        ret1570.1: {}
        '789': {}
      output:
        ret1571.1: {}
      attr:
        algorithm: mul
    aten::linear_3566:
      type: InnerProduct
      input:
        ret1571.1: {}
        '790': {}
        aten::linear_3566_bias: {}
      output:
        ret1574.1: {}
    aten::silu_5099:
      type: Swish
      input:
        ret1574.1: {}
      output:
        ret1575.1: {}
    aten::mul_629:
      type: Mul
      input:
        ret1570.1: {}
        '791': {}
      output:
        ret1576.1: {}
      attr:
        algorithm: mul
    aten::linear_3567:
      type: InnerProduct
      input:
        ret1576.1: {}
        '792': {}
        aten::linear_3567_bias: {}
      output:
        ret1579.1: {}
    aten::mul_5100:
      type: Mul
      input:
        ret1575.1: {}
        ret1579.1: {}
      output:
        ret1580.1: {}
      attr:
        algorithm: mul
    aten::mul_631:
      type: Mul
      input:
        ret1580.1: {}
        '793': {}
      output:
        ret1581.1: {}
      attr:
        algorithm: mul
    aten::linear_3568:
      type: InnerProduct
      input:
        ret1581.1: {}
        '794': {}
        aten::linear_3568_bias: {}
      output:
        ret1584.1: {}
    aten::add_3199:
      type: Add
      input:
        ret1564.1: {}
        ret1584.1: {}
      output:
        ret1585.1: {}
    aten::pow_2785:
      type: Pow
      input:
        ret1585.1: {}
        aten::pow_2785_other: {}
      output:
        ret1586.1: {}
    aten::mean_1011:
      type: ReduceMean
      input:
        ret1586.1: {}
      output:
        ret1587.1: {}
    aten::add_46:
      type: Add
      input:
        ret1587.1: {}
        '485': {}
      output:
        ret1588.1: {}
    aten::rsqrt_5104:
      type: Rsqrt
      input:
        ret1588.1: {}
      output:
        ret1589.1: {}
    aten::mul_5103:
      type: Mul
      input:
        ret1585.1: {}
        ret1589.1: {}
      output:
        ret1590.1: {}
      attr:
        algorithm: mul
    aten::mul_633:
      type: Mul
      input:
        '795': {}
        ret1590.1: {}
      output:
        ret1591.1: {}
      attr:
        algorithm: mul
    aten::size_3408:
      type: Shape
      input:
        ret1591.1: {}
      output:
        '5649': {}
      attr:
        start: 0
        end: 0
    aten::size_3200:
      type: Shape
      input:
        ret1591.1: {}
      output:
        '5651': {}
      attr:
        start: 1
        end: 1
    aten::mul_634:
      type: Mul
      input:
        ret1591.1: {}
        '796': {}
      output:
        ret1594.1: {}
      attr:
        algorithm: mul
    aten::linear_3569:
      type: InnerProduct
      input:
        ret1594.1: {}
        '797': {}
        aten::linear_3569_bias: {}
      output:
        ret1597.1: {}
    aten::view_5106:
      type: View
      input:
        ret1597.1: {}
        '5649': {}
        '5651': {}
      output:
        ret1598.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2786:
      type: Reorder
      input:
        ret1598.1: {}
      output:
        ret1599.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_636:
      type: Mul
      input:
        ret1591.1: {}
        '798': {}
      output:
        ret1600.1: {}
      attr:
        algorithm: mul
    aten::linear_3570:
      type: InnerProduct
      input:
        ret1600.1: {}
        '799': {}
        aten::linear_3570_bias: {}
      output:
        ret1603.1: {}
    aten::view_5107:
      type: View
      input:
        ret1603.1: {}
        '5649': {}
        '5651': {}
      output:
        ret1604.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2787:
      type: Reorder
      input:
        ret1604.1: {}
      output:
        ret1605.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_638:
      type: Mul
      input:
        ret1591.1: {}
        '800': {}
      output:
        ret1606.1: {}
      attr:
        algorithm: mul
    aten::linear_3571:
      type: InnerProduct
      input:
        ret1606.1: {}
        '801': {}
        aten::linear_3571_bias: {}
      output:
        ret1609.1: {}
    aten::view_5108:
      type: View
      input:
        ret1609.1: {}
        '5649': {}
        '5651': {}
      output:
        ret1610.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2788:
      type: Reorder
      input:
        ret1610.1: {}
      output:
        ret1611.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2789:
      type: Shape
      input:
        ret1605.1: {}
      output:
        '5697': {}
      attr:
        start: 2
        end: 2
    aten::size_2790:
      type: Shape
      input:
        x37.1: {}
      output:
        '5700': {}
      attr:
        start: 2
        end: 2
    aten::add_3366:
      type: Add
      input:
        '5697': {}
        '5700': {}
      output:
        seq_len18.1: {}
    aten::slice_115:
      type: Slice
      input:
        '493': {}
        seq_len18.1: {}
      output:
        '5707': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_155:
      type: Slice
      input:
        '494': {}
        seq_len18.1: {}
      output:
        '5709': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2791:
      type: Shape
      input:
        ret1599.1: {}
      output:
        '5711': {}
      attr:
        start: 2
        end: 2
    aten::add_3201:
      type: Add
      input:
        '5711': {}
        '5700': {}
      output:
        '5716': {}
    aten::slice_2792:
      type: Slice
      input:
        '5707': {}
        '5700': {}
        '5716': {}
      output:
        '5718': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2216:
      type: Slice
      input:
        '5718': {}
      output:
        '5719': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2793:
      type: Slice
      input:
        '5709': {}
        '5700': {}
        '5716': {}
      output:
        '5720': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2217:
      type: Slice
      input:
        '5720': {}
      output:
        '5721': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5109:
      type: Mul
      input:
        ret1599.1: {}
        '5719': {}
      output:
        ret1615.1: {}
      attr:
        algorithm: mul
    aten::size_2218:
      type: Shape
      input:
        ret1599.1: {}
      output:
        '5725': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_214:
      type: Div
      input:
        '5725': {}
        '495': {}
      output:
        ret1617.1: {}
      attr:
        algorithm: div
    aten::slice_2219:
      type: Slice
      input:
        ret1599.1: {}
        ret1617.1: {}
      output:
        ret1618.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2220:
      type: Slice
      input:
        ret1599.1: {}
        ret1617.1: {}
      output:
        ret1619.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5116:
      type: Neg
      input:
        ret1619.1: {}
        aten::neg_5116_mul_val: {}
      output:
        ret1620.1: {}
      attr:
        algorithm: mul
    aten::cat_2491:
      type: Concat
      input:
        ret1620.1: {}
        ret1618.1: {}
      output:
        ret1621.1: {}
      attr:
        axis: -1
    aten::mul_5113:
      type: Mul
      input:
        ret1621.1: {}
        '5721': {}
      output:
        ret1622.1: {}
      attr:
        algorithm: mul
    aten::add_3202:
      type: Add
      input:
        ret1615.1: {}
        ret1622.1: {}
      output:
        args75.1: {}
    aten::mul_5111:
      type: Mul
      input:
        ret1605.1: {}
        '5719': {}
      output:
        ret1623.1: {}
      attr:
        algorithm: mul
    aten::size_2221:
      type: Shape
      input:
        ret1605.1: {}
      output:
        '5747': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_215:
      type: Div
      input:
        '5747': {}
        '495': {}
      output:
        ret1625.1: {}
      attr:
        algorithm: div
    aten::slice_2222:
      type: Slice
      input:
        ret1605.1: {}
        ret1625.1: {}
      output:
        ret1626.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2223:
      type: Slice
      input:
        ret1605.1: {}
        ret1625.1: {}
      output:
        ret1627.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5118:
      type: Neg
      input:
        ret1627.1: {}
        aten::neg_5118_mul_val: {}
      output:
        ret1628.1: {}
      attr:
        algorithm: mul
    aten::cat_2492:
      type: Concat
      input:
        ret1628.1: {}
        ret1626.1: {}
      output:
        ret1629.1: {}
      attr:
        axis: -1
    aten::mul_5114:
      type: Mul
      input:
        ret1629.1: {}
        '5721': {}
      output:
        ret1630.1: {}
      attr:
        algorithm: mul
    aten::add_3203:
      type: Add
      input:
        ret1623.1: {}
        ret1630.1: {}
      output:
        '5765': {}
    aten::cat_2794:
      type: Concat
      input:
        x37.1: {}
        '5765': {}
      output:
        ret1631.1: {}
      attr:
        axis: 2
    aten::cat_2795:
      type: Concat
      input:
        x38.1: {}
        ret1611.1: {}
      output:
        ret1632.1: {}
      attr:
        axis: 2
    aten::transpose_2224:
      type: Reorder
      input:
        ret1631.1: {}
      output:
        ret1633.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5121:
      type: Matmul
      input:
        args75.1: {}
        ret1633.1: {}
      output:
        ret1636.1: {}
    aten::div_275:
      type: Div
      input:
        ret1636.1: {}
        '496': {}
      output:
        ret1637.1: {}
      attr:
        algorithm: div
    aten::add_3204:
      type: Add
      input:
        ret1637.1: {}
        attention_mask0.1: {}
      output:
        attn_weights38.1: {}
    aten::max_315:
      type: Max
      input:
        attn_weights38.1: {}
        '497': {}
      output:
        input38.1: {}
    aten::softmax_2428:
      type: Softmax
      input:
        input38.1: {}
      output:
        '5787': {}
      attr:
        axis: -1
    aten::matmul_5124:
      type: Matmul
      input:
        '5787': {}
        ret1632.1: {}
      output:
        ret1640.1: {}
    aten::transpose_2796:
      type: Reorder
      input:
        ret1640.1: {}
      output:
        ret1641.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5126:
      type: Reshape
      input:
        ret1641.1: {}
        '5649': {}
        '5651': {}
      output:
        ret1642.1: {}
    aten::mul_640:
      type: Mul
      input:
        ret1642.1: {}
        '802': {}
      output:
        ret1643.1: {}
      attr:
        algorithm: mul
    aten::linear_3572:
      type: InnerProduct
      input:
        ret1643.1: {}
        '803': {}
        aten::linear_3572_bias: {}
      output:
        ret1646.1: {}
    aten::add_3205:
      type: Add
      input:
        ret1585.1: {}
        ret1646.1: {}
      output:
        ret1647.1: {}
    aten::pow_2797:
      type: Pow
      input:
        ret1647.1: {}
        aten::pow_2797_other: {}
      output:
        ret1648.1: {}
    aten::mean_1012:
      type: ReduceMean
      input:
        ret1648.1: {}
      output:
        ret1649.1: {}
    aten::add_47:
      type: Add
      input:
        ret1649.1: {}
        '485': {}
      output:
        ret1650.1: {}
    aten::rsqrt_5129:
      type: Rsqrt
      input:
        ret1650.1: {}
      output:
        ret1651.1: {}
    aten::mul_5128:
      type: Mul
      input:
        ret1647.1: {}
        ret1651.1: {}
      output:
        ret1652.1: {}
      attr:
        algorithm: mul
    aten::mul_642:
      type: Mul
      input:
        '804': {}
        ret1652.1: {}
      output:
        ret1653.1: {}
      attr:
        algorithm: mul
    aten::mul_643:
      type: Mul
      input:
        ret1653.1: {}
        '805': {}
      output:
        ret1654.1: {}
      attr:
        algorithm: mul
    aten::linear_3573:
      type: InnerProduct
      input:
        ret1654.1: {}
        '806': {}
        aten::linear_3573_bias: {}
      output:
        ret1657.1: {}
    aten::silu_5131:
      type: Swish
      input:
        ret1657.1: {}
      output:
        ret1658.1: {}
    aten::mul_645:
      type: Mul
      input:
        ret1653.1: {}
        '807': {}
      output:
        ret1659.1: {}
      attr:
        algorithm: mul
    aten::linear_3574:
      type: InnerProduct
      input:
        ret1659.1: {}
        '808': {}
        aten::linear_3574_bias: {}
      output:
        ret1662.1: {}
    aten::mul_5132:
      type: Mul
      input:
        ret1658.1: {}
        ret1662.1: {}
      output:
        ret1663.1: {}
      attr:
        algorithm: mul
    aten::mul_647:
      type: Mul
      input:
        ret1663.1: {}
        '809': {}
      output:
        ret1664.1: {}
      attr:
        algorithm: mul
    aten::linear_3575:
      type: InnerProduct
      input:
        ret1664.1: {}
        '810': {}
        aten::linear_3575_bias: {}
      output:
        ret1667.1: {}
    aten::add_3206:
      type: Add
      input:
        ret1647.1: {}
        ret1667.1: {}
      output:
        ret1668.1: {}
    aten::pow_2798:
      type: Pow
      input:
        ret1668.1: {}
        aten::pow_2798_other: {}
      output:
        ret1669.1: {}
    aten::mean_1013:
      type: ReduceMean
      input:
        ret1669.1: {}
      output:
        ret1670.1: {}
    aten::add_48:
      type: Add
      input:
        ret1670.1: {}
        '485': {}
      output:
        ret1671.1: {}
    aten::rsqrt_5136:
      type: Rsqrt
      input:
        ret1671.1: {}
      output:
        ret1672.1: {}
    aten::mul_5135:
      type: Mul
      input:
        ret1668.1: {}
        ret1672.1: {}
      output:
        ret1673.1: {}
      attr:
        algorithm: mul
    aten::mul_649:
      type: Mul
      input:
        '811': {}
        ret1673.1: {}
      output:
        ret1674.1: {}
      attr:
        algorithm: mul
    aten::size_3409:
      type: Shape
      input:
        ret1674.1: {}
      output:
        '5872': {}
      attr:
        start: 0
        end: 0
    aten::size_3207:
      type: Shape
      input:
        ret1674.1: {}
      output:
        '5874': {}
      attr:
        start: 1
        end: 1
    aten::mul_650:
      type: Mul
      input:
        ret1674.1: {}
        '812': {}
      output:
        ret1677.1: {}
      attr:
        algorithm: mul
    aten::linear_3576:
      type: InnerProduct
      input:
        ret1677.1: {}
        '813': {}
        aten::linear_3576_bias: {}
      output:
        ret1680.1: {}
    aten::view_5138:
      type: View
      input:
        ret1680.1: {}
        '5872': {}
        '5874': {}
      output:
        ret1681.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2799:
      type: Reorder
      input:
        ret1681.1: {}
      output:
        ret1682.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_652:
      type: Mul
      input:
        ret1674.1: {}
        '814': {}
      output:
        ret1683.1: {}
      attr:
        algorithm: mul
    aten::linear_3577:
      type: InnerProduct
      input:
        ret1683.1: {}
        '815': {}
        aten::linear_3577_bias: {}
      output:
        ret1686.1: {}
    aten::view_5139:
      type: View
      input:
        ret1686.1: {}
        '5872': {}
        '5874': {}
      output:
        ret1687.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2800:
      type: Reorder
      input:
        ret1687.1: {}
      output:
        ret1688.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_654:
      type: Mul
      input:
        ret1674.1: {}
        '816': {}
      output:
        ret1689.1: {}
      attr:
        algorithm: mul
    aten::linear_3578:
      type: InnerProduct
      input:
        ret1689.1: {}
        '817': {}
        aten::linear_3578_bias: {}
      output:
        ret1692.1: {}
    aten::view_5140:
      type: View
      input:
        ret1692.1: {}
        '5872': {}
        '5874': {}
      output:
        ret1693.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2801:
      type: Reorder
      input:
        ret1693.1: {}
      output:
        ret1694.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2802:
      type: Shape
      input:
        ret1688.1: {}
      output:
        '5920': {}
      attr:
        start: 2
        end: 2
    aten::size_2803:
      type: Shape
      input:
        x39.1: {}
      output:
        '5923': {}
      attr:
        start: 2
        end: 2
    aten::add_3367:
      type: Add
      input:
        '5920': {}
        '5923': {}
      output:
        seq_len19.1: {}
    aten::slice_116:
      type: Slice
      input:
        '493': {}
        seq_len19.1: {}
      output:
        '5930': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_156:
      type: Slice
      input:
        '494': {}
        seq_len19.1: {}
      output:
        '5932': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2804:
      type: Shape
      input:
        ret1682.1: {}
      output:
        '5934': {}
      attr:
        start: 2
        end: 2
    aten::add_3208:
      type: Add
      input:
        '5934': {}
        '5923': {}
      output:
        '5939': {}
    aten::slice_2805:
      type: Slice
      input:
        '5930': {}
        '5923': {}
        '5939': {}
      output:
        '5941': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2225:
      type: Slice
      input:
        '5941': {}
      output:
        '5942': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2806:
      type: Slice
      input:
        '5932': {}
        '5923': {}
        '5939': {}
      output:
        '5943': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2226:
      type: Slice
      input:
        '5943': {}
      output:
        '5944': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5141:
      type: Mul
      input:
        ret1682.1: {}
        '5942': {}
      output:
        ret1698.1: {}
      attr:
        algorithm: mul
    aten::size_2227:
      type: Shape
      input:
        ret1682.1: {}
      output:
        '5948': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_216:
      type: Div
      input:
        '5948': {}
        '495': {}
      output:
        ret1700.1: {}
      attr:
        algorithm: div
    aten::slice_2228:
      type: Slice
      input:
        ret1682.1: {}
        ret1700.1: {}
      output:
        ret1701.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2229:
      type: Slice
      input:
        ret1682.1: {}
        ret1700.1: {}
      output:
        ret1702.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5148:
      type: Neg
      input:
        ret1702.1: {}
        aten::neg_5148_mul_val: {}
      output:
        ret1703.1: {}
      attr:
        algorithm: mul
    aten::cat_2493:
      type: Concat
      input:
        ret1703.1: {}
        ret1701.1: {}
      output:
        ret1704.1: {}
      attr:
        axis: -1
    aten::mul_5145:
      type: Mul
      input:
        ret1704.1: {}
        '5944': {}
      output:
        ret1705.1: {}
      attr:
        algorithm: mul
    aten::add_3209:
      type: Add
      input:
        ret1698.1: {}
        ret1705.1: {}
      output:
        args79.1: {}
    aten::mul_5143:
      type: Mul
      input:
        ret1688.1: {}
        '5942': {}
      output:
        ret1706.1: {}
      attr:
        algorithm: mul
    aten::size_2230:
      type: Shape
      input:
        ret1688.1: {}
      output:
        '5970': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_217:
      type: Div
      input:
        '5970': {}
        '495': {}
      output:
        ret1708.1: {}
      attr:
        algorithm: div
    aten::slice_2231:
      type: Slice
      input:
        ret1688.1: {}
        ret1708.1: {}
      output:
        ret1709.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2232:
      type: Slice
      input:
        ret1688.1: {}
        ret1708.1: {}
      output:
        ret1710.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5150:
      type: Neg
      input:
        ret1710.1: {}
        aten::neg_5150_mul_val: {}
      output:
        ret1711.1: {}
      attr:
        algorithm: mul
    aten::cat_2494:
      type: Concat
      input:
        ret1711.1: {}
        ret1709.1: {}
      output:
        ret1712.1: {}
      attr:
        axis: -1
    aten::mul_5146:
      type: Mul
      input:
        ret1712.1: {}
        '5944': {}
      output:
        ret1713.1: {}
      attr:
        algorithm: mul
    aten::add_3210:
      type: Add
      input:
        ret1706.1: {}
        ret1713.1: {}
      output:
        '5988': {}
    aten::cat_2807:
      type: Concat
      input:
        x39.1: {}
        '5988': {}
      output:
        ret1714.1: {}
      attr:
        axis: 2
    aten::cat_2808:
      type: Concat
      input:
        x40.1: {}
        ret1694.1: {}
      output:
        ret1715.1: {}
      attr:
        axis: 2
    aten::transpose_2233:
      type: Reorder
      input:
        ret1714.1: {}
      output:
        ret1716.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5153:
      type: Matmul
      input:
        args79.1: {}
        ret1716.1: {}
      output:
        ret1719.1: {}
    aten::div_276:
      type: Div
      input:
        ret1719.1: {}
        '496': {}
      output:
        ret1720.1: {}
      attr:
        algorithm: div
    aten::add_3211:
      type: Add
      input:
        ret1720.1: {}
        attention_mask0.1: {}
      output:
        attn_weights40.1: {}
    aten::max_316:
      type: Max
      input:
        attn_weights40.1: {}
        '497': {}
      output:
        input40.1: {}
    aten::softmax_2429:
      type: Softmax
      input:
        input40.1: {}
      output:
        '6010': {}
      attr:
        axis: -1
    aten::matmul_5156:
      type: Matmul
      input:
        '6010': {}
        ret1715.1: {}
      output:
        ret1723.1: {}
    aten::transpose_2809:
      type: Reorder
      input:
        ret1723.1: {}
      output:
        ret1724.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5158:
      type: Reshape
      input:
        ret1724.1: {}
        '5872': {}
        '5874': {}
      output:
        ret1725.1: {}
    aten::mul_656:
      type: Mul
      input:
        ret1725.1: {}
        '818': {}
      output:
        ret1726.1: {}
      attr:
        algorithm: mul
    aten::linear_3579:
      type: InnerProduct
      input:
        ret1726.1: {}
        '819': {}
        aten::linear_3579_bias: {}
      output:
        ret1729.1: {}
    aten::add_3212:
      type: Add
      input:
        ret1668.1: {}
        ret1729.1: {}
      output:
        ret1730.1: {}
    aten::pow_2810:
      type: Pow
      input:
        ret1730.1: {}
        aten::pow_2810_other: {}
      output:
        ret1731.1: {}
    aten::mean_1014:
      type: ReduceMean
      input:
        ret1731.1: {}
      output:
        ret1732.1: {}
    aten::add_49:
      type: Add
      input:
        ret1732.1: {}
        '485': {}
      output:
        ret1733.1: {}
    aten::rsqrt_5161:
      type: Rsqrt
      input:
        ret1733.1: {}
      output:
        ret1734.1: {}
    aten::mul_5160:
      type: Mul
      input:
        ret1730.1: {}
        ret1734.1: {}
      output:
        ret1735.1: {}
      attr:
        algorithm: mul
    aten::mul_658:
      type: Mul
      input:
        '820': {}
        ret1735.1: {}
      output:
        ret1736.1: {}
      attr:
        algorithm: mul
    aten::mul_659:
      type: Mul
      input:
        ret1736.1: {}
        '821': {}
      output:
        ret1737.1: {}
      attr:
        algorithm: mul
    aten::linear_3580:
      type: InnerProduct
      input:
        ret1737.1: {}
        '822': {}
        aten::linear_3580_bias: {}
      output:
        ret1740.1: {}
    aten::silu_5163:
      type: Swish
      input:
        ret1740.1: {}
      output:
        ret1741.1: {}
    aten::mul_661:
      type: Mul
      input:
        ret1736.1: {}
        '823': {}
      output:
        ret1742.1: {}
      attr:
        algorithm: mul
    aten::linear_3581:
      type: InnerProduct
      input:
        ret1742.1: {}
        '824': {}
        aten::linear_3581_bias: {}
      output:
        ret1745.1: {}
    aten::mul_5164:
      type: Mul
      input:
        ret1741.1: {}
        ret1745.1: {}
      output:
        ret1746.1: {}
      attr:
        algorithm: mul
    aten::mul_663:
      type: Mul
      input:
        ret1746.1: {}
        '825': {}
      output:
        ret1747.1: {}
      attr:
        algorithm: mul
    aten::linear_3582:
      type: InnerProduct
      input:
        ret1747.1: {}
        '826': {}
        aten::linear_3582_bias: {}
      output:
        ret1750.1: {}
    aten::add_3213:
      type: Add
      input:
        ret1730.1: {}
        ret1750.1: {}
      output:
        ret1751.1: {}
    aten::pow_2811:
      type: Pow
      input:
        ret1751.1: {}
        aten::pow_2811_other: {}
      output:
        ret1752.1: {}
    aten::mean_1015:
      type: ReduceMean
      input:
        ret1752.1: {}
      output:
        ret1753.1: {}
    aten::add_50:
      type: Add
      input:
        ret1753.1: {}
        '485': {}
      output:
        ret1754.1: {}
    aten::rsqrt_5168:
      type: Rsqrt
      input:
        ret1754.1: {}
      output:
        ret1755.1: {}
    aten::mul_5167:
      type: Mul
      input:
        ret1751.1: {}
        ret1755.1: {}
      output:
        ret1756.1: {}
      attr:
        algorithm: mul
    aten::mul_665:
      type: Mul
      input:
        '827': {}
        ret1756.1: {}
      output:
        ret1757.1: {}
      attr:
        algorithm: mul
    aten::size_3410:
      type: Shape
      input:
        ret1757.1: {}
      output:
        '6095': {}
      attr:
        start: 0
        end: 0
    aten::size_3214:
      type: Shape
      input:
        ret1757.1: {}
      output:
        '6097': {}
      attr:
        start: 1
        end: 1
    aten::mul_666:
      type: Mul
      input:
        ret1757.1: {}
        '828': {}
      output:
        ret1760.1: {}
      attr:
        algorithm: mul
    aten::linear_3583:
      type: InnerProduct
      input:
        ret1760.1: {}
        '829': {}
        aten::linear_3583_bias: {}
      output:
        ret1763.1: {}
    aten::view_5170:
      type: View
      input:
        ret1763.1: {}
        '6095': {}
        '6097': {}
      output:
        ret1764.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2812:
      type: Reorder
      input:
        ret1764.1: {}
      output:
        ret1765.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_668:
      type: Mul
      input:
        ret1757.1: {}
        '830': {}
      output:
        ret1766.1: {}
      attr:
        algorithm: mul
    aten::linear_3584:
      type: InnerProduct
      input:
        ret1766.1: {}
        '831': {}
        aten::linear_3584_bias: {}
      output:
        ret1769.1: {}
    aten::view_5171:
      type: View
      input:
        ret1769.1: {}
        '6095': {}
        '6097': {}
      output:
        ret1770.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2813:
      type: Reorder
      input:
        ret1770.1: {}
      output:
        ret1771.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_670:
      type: Mul
      input:
        ret1757.1: {}
        '832': {}
      output:
        ret1772.1: {}
      attr:
        algorithm: mul
    aten::linear_3585:
      type: InnerProduct
      input:
        ret1772.1: {}
        '833': {}
        aten::linear_3585_bias: {}
      output:
        ret1775.1: {}
    aten::view_5172:
      type: View
      input:
        ret1775.1: {}
        '6095': {}
        '6097': {}
      output:
        ret1776.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2814:
      type: Reorder
      input:
        ret1776.1: {}
      output:
        ret1777.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2815:
      type: Shape
      input:
        ret1771.1: {}
      output:
        '6143': {}
      attr:
        start: 2
        end: 2
    aten::size_2816:
      type: Shape
      input:
        x41.1: {}
      output:
        '6146': {}
      attr:
        start: 2
        end: 2
    aten::add_3368:
      type: Add
      input:
        '6143': {}
        '6146': {}
      output:
        seq_len20.1: {}
    aten::slice_117:
      type: Slice
      input:
        '493': {}
        seq_len20.1: {}
      output:
        '6153': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_157:
      type: Slice
      input:
        '494': {}
        seq_len20.1: {}
      output:
        '6155': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2817:
      type: Shape
      input:
        ret1765.1: {}
      output:
        '6157': {}
      attr:
        start: 2
        end: 2
    aten::add_3215:
      type: Add
      input:
        '6157': {}
        '6146': {}
      output:
        '6162': {}
    aten::slice_2818:
      type: Slice
      input:
        '6153': {}
        '6146': {}
        '6162': {}
      output:
        '6164': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2234:
      type: Slice
      input:
        '6164': {}
      output:
        '6165': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2819:
      type: Slice
      input:
        '6155': {}
        '6146': {}
        '6162': {}
      output:
        '6166': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2235:
      type: Slice
      input:
        '6166': {}
      output:
        '6167': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5173:
      type: Mul
      input:
        ret1765.1: {}
        '6165': {}
      output:
        ret1781.1: {}
      attr:
        algorithm: mul
    aten::size_2236:
      type: Shape
      input:
        ret1765.1: {}
      output:
        '6171': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_218:
      type: Div
      input:
        '6171': {}
        '495': {}
      output:
        ret1783.1: {}
      attr:
        algorithm: div
    aten::slice_2237:
      type: Slice
      input:
        ret1765.1: {}
        ret1783.1: {}
      output:
        ret1784.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2238:
      type: Slice
      input:
        ret1765.1: {}
        ret1783.1: {}
      output:
        ret1785.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5180:
      type: Neg
      input:
        ret1785.1: {}
        aten::neg_5180_mul_val: {}
      output:
        ret1786.1: {}
      attr:
        algorithm: mul
    aten::cat_2495:
      type: Concat
      input:
        ret1786.1: {}
        ret1784.1: {}
      output:
        ret1787.1: {}
      attr:
        axis: -1
    aten::mul_5177:
      type: Mul
      input:
        ret1787.1: {}
        '6167': {}
      output:
        ret1788.1: {}
      attr:
        algorithm: mul
    aten::add_3216:
      type: Add
      input:
        ret1781.1: {}
        ret1788.1: {}
      output:
        args83.1: {}
    aten::mul_5175:
      type: Mul
      input:
        ret1771.1: {}
        '6165': {}
      output:
        ret1789.1: {}
      attr:
        algorithm: mul
    aten::size_2239:
      type: Shape
      input:
        ret1771.1: {}
      output:
        '6193': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_219:
      type: Div
      input:
        '6193': {}
        '495': {}
      output:
        ret1791.1: {}
      attr:
        algorithm: div
    aten::slice_2240:
      type: Slice
      input:
        ret1771.1: {}
        ret1791.1: {}
      output:
        ret1792.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2241:
      type: Slice
      input:
        ret1771.1: {}
        ret1791.1: {}
      output:
        ret1793.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5182:
      type: Neg
      input:
        ret1793.1: {}
        aten::neg_5182_mul_val: {}
      output:
        ret1794.1: {}
      attr:
        algorithm: mul
    aten::cat_2496:
      type: Concat
      input:
        ret1794.1: {}
        ret1792.1: {}
      output:
        ret1795.1: {}
      attr:
        axis: -1
    aten::mul_5178:
      type: Mul
      input:
        ret1795.1: {}
        '6167': {}
      output:
        ret1796.1: {}
      attr:
        algorithm: mul
    aten::add_3217:
      type: Add
      input:
        ret1789.1: {}
        ret1796.1: {}
      output:
        '6211': {}
    aten::cat_2820:
      type: Concat
      input:
        x41.1: {}
        '6211': {}
      output:
        ret1797.1: {}
      attr:
        axis: 2
    aten::cat_2821:
      type: Concat
      input:
        x42.1: {}
        ret1777.1: {}
      output:
        ret1798.1: {}
      attr:
        axis: 2
    aten::transpose_2242:
      type: Reorder
      input:
        ret1797.1: {}
      output:
        ret1799.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5185:
      type: Matmul
      input:
        args83.1: {}
        ret1799.1: {}
      output:
        ret1802.1: {}
    aten::div_277:
      type: Div
      input:
        ret1802.1: {}
        '496': {}
      output:
        ret1803.1: {}
      attr:
        algorithm: div
    aten::add_3218:
      type: Add
      input:
        ret1803.1: {}
        attention_mask0.1: {}
      output:
        attn_weights42.1: {}
    aten::max_317:
      type: Max
      input:
        attn_weights42.1: {}
        '497': {}
      output:
        input42.1: {}
    aten::softmax_2430:
      type: Softmax
      input:
        input42.1: {}
      output:
        '6233': {}
      attr:
        axis: -1
    aten::matmul_5188:
      type: Matmul
      input:
        '6233': {}
        ret1798.1: {}
      output:
        ret1806.1: {}
    aten::transpose_2822:
      type: Reorder
      input:
        ret1806.1: {}
      output:
        ret1807.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5190:
      type: Reshape
      input:
        ret1807.1: {}
        '6095': {}
        '6097': {}
      output:
        ret1808.1: {}
    aten::mul_672:
      type: Mul
      input:
        ret1808.1: {}
        '834': {}
      output:
        ret1809.1: {}
      attr:
        algorithm: mul
    aten::linear_3586:
      type: InnerProduct
      input:
        ret1809.1: {}
        '835': {}
        aten::linear_3586_bias: {}
      output:
        ret1812.1: {}
    aten::add_3219:
      type: Add
      input:
        ret1751.1: {}
        ret1812.1: {}
      output:
        ret1813.1: {}
    aten::pow_2823:
      type: Pow
      input:
        ret1813.1: {}
        aten::pow_2823_other: {}
      output:
        ret1814.1: {}
    aten::mean_1016:
      type: ReduceMean
      input:
        ret1814.1: {}
      output:
        ret1815.1: {}
    aten::add_51:
      type: Add
      input:
        ret1815.1: {}
        '485': {}
      output:
        ret1816.1: {}
    aten::rsqrt_5193:
      type: Rsqrt
      input:
        ret1816.1: {}
      output:
        ret1817.1: {}
    aten::mul_5192:
      type: Mul
      input:
        ret1813.1: {}
        ret1817.1: {}
      output:
        ret1818.1: {}
      attr:
        algorithm: mul
    aten::mul_674:
      type: Mul
      input:
        '836': {}
        ret1818.1: {}
      output:
        ret1819.1: {}
      attr:
        algorithm: mul
    aten::mul_675:
      type: Mul
      input:
        ret1819.1: {}
        '837': {}
      output:
        ret1820.1: {}
      attr:
        algorithm: mul
    aten::linear_3587:
      type: InnerProduct
      input:
        ret1820.1: {}
        '838': {}
        aten::linear_3587_bias: {}
      output:
        ret1823.1: {}
    aten::silu_5195:
      type: Swish
      input:
        ret1823.1: {}
      output:
        ret1824.1: {}
    aten::mul_677:
      type: Mul
      input:
        ret1819.1: {}
        '839': {}
      output:
        ret1825.1: {}
      attr:
        algorithm: mul
    aten::linear_3588:
      type: InnerProduct
      input:
        ret1825.1: {}
        '840': {}
        aten::linear_3588_bias: {}
      output:
        ret1828.1: {}
    aten::mul_5196:
      type: Mul
      input:
        ret1824.1: {}
        ret1828.1: {}
      output:
        ret1829.1: {}
      attr:
        algorithm: mul
    aten::mul_679:
      type: Mul
      input:
        ret1829.1: {}
        '841': {}
      output:
        ret1830.1: {}
      attr:
        algorithm: mul
    aten::linear_3589:
      type: InnerProduct
      input:
        ret1830.1: {}
        '842': {}
        aten::linear_3589_bias: {}
      output:
        ret1833.1: {}
    aten::add_3220:
      type: Add
      input:
        ret1813.1: {}
        ret1833.1: {}
      output:
        ret1834.1: {}
    aten::pow_2824:
      type: Pow
      input:
        ret1834.1: {}
        aten::pow_2824_other: {}
      output:
        ret1835.1: {}
    aten::mean_1017:
      type: ReduceMean
      input:
        ret1835.1: {}
      output:
        ret1836.1: {}
    aten::add_52:
      type: Add
      input:
        ret1836.1: {}
        '485': {}
      output:
        ret1837.1: {}
    aten::rsqrt_5200:
      type: Rsqrt
      input:
        ret1837.1: {}
      output:
        ret1838.1: {}
    aten::mul_5199:
      type: Mul
      input:
        ret1834.1: {}
        ret1838.1: {}
      output:
        ret1839.1: {}
      attr:
        algorithm: mul
    aten::mul_681:
      type: Mul
      input:
        '843': {}
        ret1839.1: {}
      output:
        ret1840.1: {}
      attr:
        algorithm: mul
    aten::size_3411:
      type: Shape
      input:
        ret1840.1: {}
      output:
        '6318': {}
      attr:
        start: 0
        end: 0
    aten::size_3221:
      type: Shape
      input:
        ret1840.1: {}
      output:
        '6320': {}
      attr:
        start: 1
        end: 1
    aten::mul_682:
      type: Mul
      input:
        ret1840.1: {}
        '844': {}
      output:
        ret1843.1: {}
      attr:
        algorithm: mul
    aten::linear_3590:
      type: InnerProduct
      input:
        ret1843.1: {}
        '845': {}
        aten::linear_3590_bias: {}
      output:
        ret1846.1: {}
    aten::view_5202:
      type: View
      input:
        ret1846.1: {}
        '6318': {}
        '6320': {}
      output:
        ret1847.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2825:
      type: Reorder
      input:
        ret1847.1: {}
      output:
        ret1848.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_684:
      type: Mul
      input:
        ret1840.1: {}
        '846': {}
      output:
        ret1849.1: {}
      attr:
        algorithm: mul
    aten::linear_3591:
      type: InnerProduct
      input:
        ret1849.1: {}
        '847': {}
        aten::linear_3591_bias: {}
      output:
        ret1852.1: {}
    aten::view_5203:
      type: View
      input:
        ret1852.1: {}
        '6318': {}
        '6320': {}
      output:
        ret1853.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2826:
      type: Reorder
      input:
        ret1853.1: {}
      output:
        ret1854.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_686:
      type: Mul
      input:
        ret1840.1: {}
        '848': {}
      output:
        ret1855.1: {}
      attr:
        algorithm: mul
    aten::linear_3592:
      type: InnerProduct
      input:
        ret1855.1: {}
        '849': {}
        aten::linear_3592_bias: {}
      output:
        ret1858.1: {}
    aten::view_5204:
      type: View
      input:
        ret1858.1: {}
        '6318': {}
        '6320': {}
      output:
        ret1859.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2827:
      type: Reorder
      input:
        ret1859.1: {}
      output:
        ret1860.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2828:
      type: Shape
      input:
        ret1854.1: {}
      output:
        '6366': {}
      attr:
        start: 2
        end: 2
    aten::size_2829:
      type: Shape
      input:
        x43.1: {}
      output:
        '6369': {}
      attr:
        start: 2
        end: 2
    aten::add_3369:
      type: Add
      input:
        '6366': {}
        '6369': {}
      output:
        seq_len21.1: {}
    aten::slice_118:
      type: Slice
      input:
        '493': {}
        seq_len21.1: {}
      output:
        '6376': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_158:
      type: Slice
      input:
        '494': {}
        seq_len21.1: {}
      output:
        '6378': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2830:
      type: Shape
      input:
        ret1848.1: {}
      output:
        '6380': {}
      attr:
        start: 2
        end: 2
    aten::add_3222:
      type: Add
      input:
        '6380': {}
        '6369': {}
      output:
        '6385': {}
    aten::slice_2831:
      type: Slice
      input:
        '6376': {}
        '6369': {}
        '6385': {}
      output:
        '6387': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2243:
      type: Slice
      input:
        '6387': {}
      output:
        '6388': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2832:
      type: Slice
      input:
        '6378': {}
        '6369': {}
        '6385': {}
      output:
        '6389': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2244:
      type: Slice
      input:
        '6389': {}
      output:
        '6390': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5205:
      type: Mul
      input:
        ret1848.1: {}
        '6388': {}
      output:
        ret1864.1: {}
      attr:
        algorithm: mul
    aten::size_2245:
      type: Shape
      input:
        ret1848.1: {}
      output:
        '6394': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_220:
      type: Div
      input:
        '6394': {}
        '495': {}
      output:
        ret1866.1: {}
      attr:
        algorithm: div
    aten::slice_2246:
      type: Slice
      input:
        ret1848.1: {}
        ret1866.1: {}
      output:
        ret1867.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2247:
      type: Slice
      input:
        ret1848.1: {}
        ret1866.1: {}
      output:
        ret1868.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5212:
      type: Neg
      input:
        ret1868.1: {}
        aten::neg_5212_mul_val: {}
      output:
        ret1869.1: {}
      attr:
        algorithm: mul
    aten::cat_2497:
      type: Concat
      input:
        ret1869.1: {}
        ret1867.1: {}
      output:
        ret1870.1: {}
      attr:
        axis: -1
    aten::mul_5209:
      type: Mul
      input:
        ret1870.1: {}
        '6390': {}
      output:
        ret1871.1: {}
      attr:
        algorithm: mul
    aten::add_3223:
      type: Add
      input:
        ret1864.1: {}
        ret1871.1: {}
      output:
        args87.1: {}
    aten::mul_5207:
      type: Mul
      input:
        ret1854.1: {}
        '6388': {}
      output:
        ret1872.1: {}
      attr:
        algorithm: mul
    aten::size_2248:
      type: Shape
      input:
        ret1854.1: {}
      output:
        '6416': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_221:
      type: Div
      input:
        '6416': {}
        '495': {}
      output:
        ret1874.1: {}
      attr:
        algorithm: div
    aten::slice_2249:
      type: Slice
      input:
        ret1854.1: {}
        ret1874.1: {}
      output:
        ret1875.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2250:
      type: Slice
      input:
        ret1854.1: {}
        ret1874.1: {}
      output:
        ret1876.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5214:
      type: Neg
      input:
        ret1876.1: {}
        aten::neg_5214_mul_val: {}
      output:
        ret1877.1: {}
      attr:
        algorithm: mul
    aten::cat_2498:
      type: Concat
      input:
        ret1877.1: {}
        ret1875.1: {}
      output:
        ret1878.1: {}
      attr:
        axis: -1
    aten::mul_5210:
      type: Mul
      input:
        ret1878.1: {}
        '6390': {}
      output:
        ret1879.1: {}
      attr:
        algorithm: mul
    aten::add_3224:
      type: Add
      input:
        ret1872.1: {}
        ret1879.1: {}
      output:
        '6434': {}
    aten::cat_2833:
      type: Concat
      input:
        x43.1: {}
        '6434': {}
      output:
        ret1880.1: {}
      attr:
        axis: 2
    aten::cat_2834:
      type: Concat
      input:
        x44.1: {}
        ret1860.1: {}
      output:
        ret1881.1: {}
      attr:
        axis: 2
    aten::transpose_2251:
      type: Reorder
      input:
        ret1880.1: {}
      output:
        ret1882.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5217:
      type: Matmul
      input:
        args87.1: {}
        ret1882.1: {}
      output:
        ret1885.1: {}
    aten::div_278:
      type: Div
      input:
        ret1885.1: {}
        '496': {}
      output:
        ret1886.1: {}
      attr:
        algorithm: div
    aten::add_3225:
      type: Add
      input:
        ret1886.1: {}
        attention_mask0.1: {}
      output:
        attn_weights44.1: {}
    aten::max_318:
      type: Max
      input:
        attn_weights44.1: {}
        '497': {}
      output:
        input44.1: {}
    aten::softmax_2431:
      type: Softmax
      input:
        input44.1: {}
      output:
        '6456': {}
      attr:
        axis: -1
    aten::matmul_5220:
      type: Matmul
      input:
        '6456': {}
        ret1881.1: {}
      output:
        ret1889.1: {}
    aten::transpose_2835:
      type: Reorder
      input:
        ret1889.1: {}
      output:
        ret1890.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5222:
      type: Reshape
      input:
        ret1890.1: {}
        '6318': {}
        '6320': {}
      output:
        ret1891.1: {}
    aten::mul_688:
      type: Mul
      input:
        ret1891.1: {}
        '850': {}
      output:
        ret1892.1: {}
      attr:
        algorithm: mul
    aten::linear_3593:
      type: InnerProduct
      input:
        ret1892.1: {}
        '851': {}
        aten::linear_3593_bias: {}
      output:
        ret1895.1: {}
    aten::add_3226:
      type: Add
      input:
        ret1834.1: {}
        ret1895.1: {}
      output:
        ret1896.1: {}
    aten::pow_2836:
      type: Pow
      input:
        ret1896.1: {}
        aten::pow_2836_other: {}
      output:
        ret1897.1: {}
    aten::mean_1018:
      type: ReduceMean
      input:
        ret1897.1: {}
      output:
        ret1898.1: {}
    aten::add_53:
      type: Add
      input:
        ret1898.1: {}
        '485': {}
      output:
        ret1899.1: {}
    aten::rsqrt_5225:
      type: Rsqrt
      input:
        ret1899.1: {}
      output:
        ret1900.1: {}
    aten::mul_5224:
      type: Mul
      input:
        ret1896.1: {}
        ret1900.1: {}
      output:
        ret1901.1: {}
      attr:
        algorithm: mul
    aten::mul_690:
      type: Mul
      input:
        '852': {}
        ret1901.1: {}
      output:
        ret1902.1: {}
      attr:
        algorithm: mul
    aten::mul_691:
      type: Mul
      input:
        ret1902.1: {}
        '853': {}
      output:
        ret1903.1: {}
      attr:
        algorithm: mul
    aten::linear_3594:
      type: InnerProduct
      input:
        ret1903.1: {}
        '854': {}
        aten::linear_3594_bias: {}
      output:
        ret1906.1: {}
    aten::silu_5227:
      type: Swish
      input:
        ret1906.1: {}
      output:
        ret1907.1: {}
    aten::mul_693:
      type: Mul
      input:
        ret1902.1: {}
        '855': {}
      output:
        ret1908.1: {}
      attr:
        algorithm: mul
    aten::linear_3595:
      type: InnerProduct
      input:
        ret1908.1: {}
        '856': {}
        aten::linear_3595_bias: {}
      output:
        ret1911.1: {}
    aten::mul_5228:
      type: Mul
      input:
        ret1907.1: {}
        ret1911.1: {}
      output:
        ret1912.1: {}
      attr:
        algorithm: mul
    aten::mul_695:
      type: Mul
      input:
        ret1912.1: {}
        '857': {}
      output:
        ret1913.1: {}
      attr:
        algorithm: mul
    aten::linear_3596:
      type: InnerProduct
      input:
        ret1913.1: {}
        '858': {}
        aten::linear_3596_bias: {}
      output:
        ret1916.1: {}
    aten::add_3227:
      type: Add
      input:
        ret1896.1: {}
        ret1916.1: {}
      output:
        ret1917.1: {}
    aten::pow_2837:
      type: Pow
      input:
        ret1917.1: {}
        aten::pow_2837_other: {}
      output:
        ret1918.1: {}
    aten::mean_1019:
      type: ReduceMean
      input:
        ret1918.1: {}
      output:
        ret1919.1: {}
    aten::add_54:
      type: Add
      input:
        ret1919.1: {}
        '485': {}
      output:
        ret1920.1: {}
    aten::rsqrt_5232:
      type: Rsqrt
      input:
        ret1920.1: {}
      output:
        ret1921.1: {}
    aten::mul_5231:
      type: Mul
      input:
        ret1917.1: {}
        ret1921.1: {}
      output:
        ret1922.1: {}
      attr:
        algorithm: mul
    aten::mul_697:
      type: Mul
      input:
        '859': {}
        ret1922.1: {}
      output:
        ret1923.1: {}
      attr:
        algorithm: mul
    aten::size_3412:
      type: Shape
      input:
        ret1923.1: {}
      output:
        '6541': {}
      attr:
        start: 0
        end: 0
    aten::size_3228:
      type: Shape
      input:
        ret1923.1: {}
      output:
        '6543': {}
      attr:
        start: 1
        end: 1
    aten::mul_698:
      type: Mul
      input:
        ret1923.1: {}
        '860': {}
      output:
        ret1926.1: {}
      attr:
        algorithm: mul
    aten::linear_3597:
      type: InnerProduct
      input:
        ret1926.1: {}
        '861': {}
        aten::linear_3597_bias: {}
      output:
        ret1929.1: {}
    aten::view_5234:
      type: View
      input:
        ret1929.1: {}
        '6541': {}
        '6543': {}
      output:
        ret1930.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2838:
      type: Reorder
      input:
        ret1930.1: {}
      output:
        ret1931.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_700:
      type: Mul
      input:
        ret1923.1: {}
        '862': {}
      output:
        ret1932.1: {}
      attr:
        algorithm: mul
    aten::linear_3598:
      type: InnerProduct
      input:
        ret1932.1: {}
        '863': {}
        aten::linear_3598_bias: {}
      output:
        ret1935.1: {}
    aten::view_5235:
      type: View
      input:
        ret1935.1: {}
        '6541': {}
        '6543': {}
      output:
        ret1936.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2839:
      type: Reorder
      input:
        ret1936.1: {}
      output:
        ret1937.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_702:
      type: Mul
      input:
        ret1923.1: {}
        '864': {}
      output:
        ret1938.1: {}
      attr:
        algorithm: mul
    aten::linear_3599:
      type: InnerProduct
      input:
        ret1938.1: {}
        '865': {}
        aten::linear_3599_bias: {}
      output:
        ret1941.1: {}
    aten::view_5236:
      type: View
      input:
        ret1941.1: {}
        '6541': {}
        '6543': {}
      output:
        ret1942.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2840:
      type: Reorder
      input:
        ret1942.1: {}
      output:
        ret1943.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2841:
      type: Shape
      input:
        ret1937.1: {}
      output:
        '6589': {}
      attr:
        start: 2
        end: 2
    aten::size_2842:
      type: Shape
      input:
        x45.1: {}
      output:
        '6592': {}
      attr:
        start: 2
        end: 2
    aten::add_3370:
      type: Add
      input:
        '6589': {}
        '6592': {}
      output:
        seq_len22.1: {}
    aten::slice_119:
      type: Slice
      input:
        '493': {}
        seq_len22.1: {}
      output:
        '6599': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_159:
      type: Slice
      input:
        '494': {}
        seq_len22.1: {}
      output:
        '6601': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2843:
      type: Shape
      input:
        ret1931.1: {}
      output:
        '6603': {}
      attr:
        start: 2
        end: 2
    aten::add_3229:
      type: Add
      input:
        '6603': {}
        '6592': {}
      output:
        '6608': {}
    aten::slice_2844:
      type: Slice
      input:
        '6599': {}
        '6592': {}
        '6608': {}
      output:
        '6610': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2252:
      type: Slice
      input:
        '6610': {}
      output:
        '6611': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2845:
      type: Slice
      input:
        '6601': {}
        '6592': {}
        '6608': {}
      output:
        '6612': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2253:
      type: Slice
      input:
        '6612': {}
      output:
        '6613': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5237:
      type: Mul
      input:
        ret1931.1: {}
        '6611': {}
      output:
        ret1947.1: {}
      attr:
        algorithm: mul
    aten::size_2254:
      type: Shape
      input:
        ret1931.1: {}
      output:
        '6617': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_222:
      type: Div
      input:
        '6617': {}
        '495': {}
      output:
        ret1949.1: {}
      attr:
        algorithm: div
    aten::slice_2255:
      type: Slice
      input:
        ret1931.1: {}
        ret1949.1: {}
      output:
        ret1950.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2256:
      type: Slice
      input:
        ret1931.1: {}
        ret1949.1: {}
      output:
        ret1951.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5244:
      type: Neg
      input:
        ret1951.1: {}
        aten::neg_5244_mul_val: {}
      output:
        ret1952.1: {}
      attr:
        algorithm: mul
    aten::cat_2499:
      type: Concat
      input:
        ret1952.1: {}
        ret1950.1: {}
      output:
        ret1953.1: {}
      attr:
        axis: -1
    aten::mul_5241:
      type: Mul
      input:
        ret1953.1: {}
        '6613': {}
      output:
        ret1954.1: {}
      attr:
        algorithm: mul
    aten::add_3230:
      type: Add
      input:
        ret1947.1: {}
        ret1954.1: {}
      output:
        args91.1: {}
    aten::mul_5239:
      type: Mul
      input:
        ret1937.1: {}
        '6611': {}
      output:
        ret1955.1: {}
      attr:
        algorithm: mul
    aten::size_2257:
      type: Shape
      input:
        ret1937.1: {}
      output:
        '6639': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_223:
      type: Div
      input:
        '6639': {}
        '495': {}
      output:
        ret1957.1: {}
      attr:
        algorithm: div
    aten::slice_2258:
      type: Slice
      input:
        ret1937.1: {}
        ret1957.1: {}
      output:
        ret1958.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2259:
      type: Slice
      input:
        ret1937.1: {}
        ret1957.1: {}
      output:
        ret1959.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5246:
      type: Neg
      input:
        ret1959.1: {}
        aten::neg_5246_mul_val: {}
      output:
        ret1960.1: {}
      attr:
        algorithm: mul
    aten::cat_2500:
      type: Concat
      input:
        ret1960.1: {}
        ret1958.1: {}
      output:
        ret1961.1: {}
      attr:
        axis: -1
    aten::mul_5242:
      type: Mul
      input:
        ret1961.1: {}
        '6613': {}
      output:
        ret1962.1: {}
      attr:
        algorithm: mul
    aten::add_3231:
      type: Add
      input:
        ret1955.1: {}
        ret1962.1: {}
      output:
        '6657': {}
    aten::cat_2846:
      type: Concat
      input:
        x45.1: {}
        '6657': {}
      output:
        ret1963.1: {}
      attr:
        axis: 2
    aten::cat_2847:
      type: Concat
      input:
        x46.1: {}
        ret1943.1: {}
      output:
        ret1964.1: {}
      attr:
        axis: 2
    aten::transpose_2260:
      type: Reorder
      input:
        ret1963.1: {}
      output:
        ret1965.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5249:
      type: Matmul
      input:
        args91.1: {}
        ret1965.1: {}
      output:
        ret1968.1: {}
    aten::div_279:
      type: Div
      input:
        ret1968.1: {}
        '496': {}
      output:
        ret1969.1: {}
      attr:
        algorithm: div
    aten::add_3232:
      type: Add
      input:
        ret1969.1: {}
        attention_mask0.1: {}
      output:
        attn_weights46.1: {}
    aten::max_319:
      type: Max
      input:
        attn_weights46.1: {}
        '497': {}
      output:
        input46.1: {}
    aten::softmax_2432:
      type: Softmax
      input:
        input46.1: {}
      output:
        '6679': {}
      attr:
        axis: -1
    aten::matmul_5252:
      type: Matmul
      input:
        '6679': {}
        ret1964.1: {}
      output:
        ret1972.1: {}
    aten::transpose_2848:
      type: Reorder
      input:
        ret1972.1: {}
      output:
        ret1973.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5254:
      type: Reshape
      input:
        ret1973.1: {}
        '6541': {}
        '6543': {}
      output:
        ret1974.1: {}
    aten::mul_704:
      type: Mul
      input:
        ret1974.1: {}
        '866': {}
      output:
        ret1975.1: {}
      attr:
        algorithm: mul
    aten::linear_3600:
      type: InnerProduct
      input:
        ret1975.1: {}
        '867': {}
        aten::linear_3600_bias: {}
      output:
        ret1978.1: {}
    aten::add_3233:
      type: Add
      input:
        ret1917.1: {}
        ret1978.1: {}
      output:
        ret1979.1: {}
    aten::pow_2849:
      type: Pow
      input:
        ret1979.1: {}
        aten::pow_2849_other: {}
      output:
        ret1980.1: {}
    aten::mean_1020:
      type: ReduceMean
      input:
        ret1980.1: {}
      output:
        ret1981.1: {}
    aten::add_55:
      type: Add
      input:
        ret1981.1: {}
        '485': {}
      output:
        ret1982.1: {}
    aten::rsqrt_5257:
      type: Rsqrt
      input:
        ret1982.1: {}
      output:
        ret1983.1: {}
    aten::mul_5256:
      type: Mul
      input:
        ret1979.1: {}
        ret1983.1: {}
      output:
        ret1984.1: {}
      attr:
        algorithm: mul
    aten::mul_706:
      type: Mul
      input:
        '868': {}
        ret1984.1: {}
      output:
        ret1985.1: {}
      attr:
        algorithm: mul
    aten::mul_707:
      type: Mul
      input:
        ret1985.1: {}
        '869': {}
      output:
        ret1986.1: {}
      attr:
        algorithm: mul
    aten::linear_3601:
      type: InnerProduct
      input:
        ret1986.1: {}
        '870': {}
        aten::linear_3601_bias: {}
      output:
        ret1989.1: {}
    aten::silu_5259:
      type: Swish
      input:
        ret1989.1: {}
      output:
        ret1990.1: {}
    aten::mul_709:
      type: Mul
      input:
        ret1985.1: {}
        '871': {}
      output:
        ret1991.1: {}
      attr:
        algorithm: mul
    aten::linear_3602:
      type: InnerProduct
      input:
        ret1991.1: {}
        '872': {}
        aten::linear_3602_bias: {}
      output:
        ret1994.1: {}
    aten::mul_5260:
      type: Mul
      input:
        ret1990.1: {}
        ret1994.1: {}
      output:
        ret1995.1: {}
      attr:
        algorithm: mul
    aten::mul_711:
      type: Mul
      input:
        ret1995.1: {}
        '873': {}
      output:
        ret1996.1: {}
      attr:
        algorithm: mul
    aten::linear_3603:
      type: InnerProduct
      input:
        ret1996.1: {}
        '874': {}
        aten::linear_3603_bias: {}
      output:
        ret1999.1: {}
    aten::add_3234:
      type: Add
      input:
        ret1979.1: {}
        ret1999.1: {}
      output:
        ret2000.1: {}
    aten::pow_2850:
      type: Pow
      input:
        ret2000.1: {}
        aten::pow_2850_other: {}
      output:
        ret2001.1: {}
    aten::mean_1021:
      type: ReduceMean
      input:
        ret2001.1: {}
      output:
        ret2002.1: {}
    aten::add_56:
      type: Add
      input:
        ret2002.1: {}
        '485': {}
      output:
        ret2003.1: {}
    aten::rsqrt_5264:
      type: Rsqrt
      input:
        ret2003.1: {}
      output:
        ret2004.1: {}
    aten::mul_5263:
      type: Mul
      input:
        ret2000.1: {}
        ret2004.1: {}
      output:
        ret2005.1: {}
      attr:
        algorithm: mul
    aten::mul_713:
      type: Mul
      input:
        '875': {}
        ret2005.1: {}
      output:
        ret2006.1: {}
      attr:
        algorithm: mul
    aten::size_3413:
      type: Shape
      input:
        ret2006.1: {}
      output:
        '6764': {}
      attr:
        start: 0
        end: 0
    aten::size_3235:
      type: Shape
      input:
        ret2006.1: {}
      output:
        '6766': {}
      attr:
        start: 1
        end: 1
    aten::mul_714:
      type: Mul
      input:
        ret2006.1: {}
        '876': {}
      output:
        ret2009.1: {}
      attr:
        algorithm: mul
    aten::linear_3604:
      type: InnerProduct
      input:
        ret2009.1: {}
        '877': {}
        aten::linear_3604_bias: {}
      output:
        ret2012.1: {}
    aten::view_5266:
      type: View
      input:
        ret2012.1: {}
        '6764': {}
        '6766': {}
      output:
        ret2013.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2851:
      type: Reorder
      input:
        ret2013.1: {}
      output:
        ret2014.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_716:
      type: Mul
      input:
        ret2006.1: {}
        '878': {}
      output:
        ret2015.1: {}
      attr:
        algorithm: mul
    aten::linear_3605:
      type: InnerProduct
      input:
        ret2015.1: {}
        '879': {}
        aten::linear_3605_bias: {}
      output:
        ret2018.1: {}
    aten::view_5267:
      type: View
      input:
        ret2018.1: {}
        '6764': {}
        '6766': {}
      output:
        ret2019.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2852:
      type: Reorder
      input:
        ret2019.1: {}
      output:
        ret2020.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_718:
      type: Mul
      input:
        ret2006.1: {}
        '880': {}
      output:
        ret2021.1: {}
      attr:
        algorithm: mul
    aten::linear_3606:
      type: InnerProduct
      input:
        ret2021.1: {}
        '881': {}
        aten::linear_3606_bias: {}
      output:
        ret2024.1: {}
    aten::view_5268:
      type: View
      input:
        ret2024.1: {}
        '6764': {}
        '6766': {}
      output:
        ret2025.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2853:
      type: Reorder
      input:
        ret2025.1: {}
      output:
        ret2026.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2854:
      type: Shape
      input:
        ret2020.1: {}
      output:
        '6812': {}
      attr:
        start: 2
        end: 2
    aten::size_2855:
      type: Shape
      input:
        x47.1: {}
      output:
        '6815': {}
      attr:
        start: 2
        end: 2
    aten::add_3371:
      type: Add
      input:
        '6812': {}
        '6815': {}
      output:
        seq_len23.1: {}
    aten::slice_120:
      type: Slice
      input:
        '493': {}
        seq_len23.1: {}
      output:
        '6822': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_160:
      type: Slice
      input:
        '494': {}
        seq_len23.1: {}
      output:
        '6824': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2856:
      type: Shape
      input:
        ret2014.1: {}
      output:
        '6826': {}
      attr:
        start: 2
        end: 2
    aten::add_3236:
      type: Add
      input:
        '6826': {}
        '6815': {}
      output:
        '6831': {}
    aten::slice_2857:
      type: Slice
      input:
        '6822': {}
        '6815': {}
        '6831': {}
      output:
        '6833': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2261:
      type: Slice
      input:
        '6833': {}
      output:
        '6834': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2858:
      type: Slice
      input:
        '6824': {}
        '6815': {}
        '6831': {}
      output:
        '6835': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2262:
      type: Slice
      input:
        '6835': {}
      output:
        '6836': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5269:
      type: Mul
      input:
        ret2014.1: {}
        '6834': {}
      output:
        ret2030.1: {}
      attr:
        algorithm: mul
    aten::size_2263:
      type: Shape
      input:
        ret2014.1: {}
      output:
        '6840': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_224:
      type: Div
      input:
        '6840': {}
        '495': {}
      output:
        ret2032.1: {}
      attr:
        algorithm: div
    aten::slice_2264:
      type: Slice
      input:
        ret2014.1: {}
        ret2032.1: {}
      output:
        ret2033.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2265:
      type: Slice
      input:
        ret2014.1: {}
        ret2032.1: {}
      output:
        ret2034.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5276:
      type: Neg
      input:
        ret2034.1: {}
        aten::neg_5276_mul_val: {}
      output:
        ret2035.1: {}
      attr:
        algorithm: mul
    aten::cat_2501:
      type: Concat
      input:
        ret2035.1: {}
        ret2033.1: {}
      output:
        ret2036.1: {}
      attr:
        axis: -1
    aten::mul_5273:
      type: Mul
      input:
        ret2036.1: {}
        '6836': {}
      output:
        ret2037.1: {}
      attr:
        algorithm: mul
    aten::add_3237:
      type: Add
      input:
        ret2030.1: {}
        ret2037.1: {}
      output:
        args95.1: {}
    aten::mul_5271:
      type: Mul
      input:
        ret2020.1: {}
        '6834': {}
      output:
        ret2038.1: {}
      attr:
        algorithm: mul
    aten::size_2266:
      type: Shape
      input:
        ret2020.1: {}
      output:
        '6862': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_225:
      type: Div
      input:
        '6862': {}
        '495': {}
      output:
        ret2040.1: {}
      attr:
        algorithm: div
    aten::slice_2267:
      type: Slice
      input:
        ret2020.1: {}
        ret2040.1: {}
      output:
        ret2041.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2268:
      type: Slice
      input:
        ret2020.1: {}
        ret2040.1: {}
      output:
        ret2042.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5278:
      type: Neg
      input:
        ret2042.1: {}
        aten::neg_5278_mul_val: {}
      output:
        ret2043.1: {}
      attr:
        algorithm: mul
    aten::cat_2502:
      type: Concat
      input:
        ret2043.1: {}
        ret2041.1: {}
      output:
        ret2044.1: {}
      attr:
        axis: -1
    aten::mul_5274:
      type: Mul
      input:
        ret2044.1: {}
        '6836': {}
      output:
        ret2045.1: {}
      attr:
        algorithm: mul
    aten::add_3238:
      type: Add
      input:
        ret2038.1: {}
        ret2045.1: {}
      output:
        '6880': {}
    aten::cat_2859:
      type: Concat
      input:
        x47.1: {}
        '6880': {}
      output:
        ret2046.1: {}
      attr:
        axis: 2
    aten::cat_2860:
      type: Concat
      input:
        x48.1: {}
        ret2026.1: {}
      output:
        ret2047.1: {}
      attr:
        axis: 2
    aten::transpose_2269:
      type: Reorder
      input:
        ret2046.1: {}
      output:
        ret2048.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5281:
      type: Matmul
      input:
        args95.1: {}
        ret2048.1: {}
      output:
        ret2051.1: {}
    aten::div_280:
      type: Div
      input:
        ret2051.1: {}
        '496': {}
      output:
        ret2052.1: {}
      attr:
        algorithm: div
    aten::add_3239:
      type: Add
      input:
        ret2052.1: {}
        attention_mask0.1: {}
      output:
        attn_weights48.1: {}
    aten::max_320:
      type: Max
      input:
        attn_weights48.1: {}
        '497': {}
      output:
        input48.1: {}
    aten::softmax_2433:
      type: Softmax
      input:
        input48.1: {}
      output:
        '6902': {}
      attr:
        axis: -1
    aten::matmul_5284:
      type: Matmul
      input:
        '6902': {}
        ret2047.1: {}
      output:
        ret2055.1: {}
    aten::transpose_2861:
      type: Reorder
      input:
        ret2055.1: {}
      output:
        ret2056.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5286:
      type: Reshape
      input:
        ret2056.1: {}
        '6764': {}
        '6766': {}
      output:
        ret2057.1: {}
    aten::mul_720:
      type: Mul
      input:
        ret2057.1: {}
        '882': {}
      output:
        ret2058.1: {}
      attr:
        algorithm: mul
    aten::linear_3607:
      type: InnerProduct
      input:
        ret2058.1: {}
        '883': {}
        aten::linear_3607_bias: {}
      output:
        ret2061.1: {}
    aten::add_3240:
      type: Add
      input:
        ret2000.1: {}
        ret2061.1: {}
      output:
        ret2062.1: {}
    aten::pow_2862:
      type: Pow
      input:
        ret2062.1: {}
        aten::pow_2862_other: {}
      output:
        ret2063.1: {}
    aten::mean_1022:
      type: ReduceMean
      input:
        ret2063.1: {}
      output:
        ret2064.1: {}
    aten::add_57:
      type: Add
      input:
        ret2064.1: {}
        '485': {}
      output:
        ret2065.1: {}
    aten::rsqrt_5289:
      type: Rsqrt
      input:
        ret2065.1: {}
      output:
        ret2066.1: {}
    aten::mul_5288:
      type: Mul
      input:
        ret2062.1: {}
        ret2066.1: {}
      output:
        ret2067.1: {}
      attr:
        algorithm: mul
    aten::mul_722:
      type: Mul
      input:
        '884': {}
        ret2067.1: {}
      output:
        ret2068.1: {}
      attr:
        algorithm: mul
    aten::mul_723:
      type: Mul
      input:
        ret2068.1: {}
        '885': {}
      output:
        ret2069.1: {}
      attr:
        algorithm: mul
    aten::linear_3608:
      type: InnerProduct
      input:
        ret2069.1: {}
        '886': {}
        aten::linear_3608_bias: {}
      output:
        ret2072.1: {}
    aten::silu_5291:
      type: Swish
      input:
        ret2072.1: {}
      output:
        ret2073.1: {}
    aten::mul_725:
      type: Mul
      input:
        ret2068.1: {}
        '887': {}
      output:
        ret2074.1: {}
      attr:
        algorithm: mul
    aten::linear_3609:
      type: InnerProduct
      input:
        ret2074.1: {}
        '888': {}
        aten::linear_3609_bias: {}
      output:
        ret2077.1: {}
    aten::mul_5292:
      type: Mul
      input:
        ret2073.1: {}
        ret2077.1: {}
      output:
        ret2078.1: {}
      attr:
        algorithm: mul
    aten::mul_727:
      type: Mul
      input:
        ret2078.1: {}
        '889': {}
      output:
        ret2079.1: {}
      attr:
        algorithm: mul
    aten::linear_3610:
      type: InnerProduct
      input:
        ret2079.1: {}
        '890': {}
        aten::linear_3610_bias: {}
      output:
        ret2082.1: {}
    aten::add_3241:
      type: Add
      input:
        ret2062.1: {}
        ret2082.1: {}
      output:
        ret2083.1: {}
    aten::pow_2863:
      type: Pow
      input:
        ret2083.1: {}
        aten::pow_2863_other: {}
      output:
        ret2084.1: {}
    aten::mean_1023:
      type: ReduceMean
      input:
        ret2084.1: {}
      output:
        ret2085.1: {}
    aten::add_58:
      type: Add
      input:
        ret2085.1: {}
        '485': {}
      output:
        ret2086.1: {}
    aten::rsqrt_5296:
      type: Rsqrt
      input:
        ret2086.1: {}
      output:
        ret2087.1: {}
    aten::mul_5295:
      type: Mul
      input:
        ret2083.1: {}
        ret2087.1: {}
      output:
        ret2088.1: {}
      attr:
        algorithm: mul
    aten::mul_729:
      type: Mul
      input:
        '891': {}
        ret2088.1: {}
      output:
        ret2089.1: {}
      attr:
        algorithm: mul
    aten::size_3414:
      type: Shape
      input:
        ret2089.1: {}
      output:
        '6987': {}
      attr:
        start: 0
        end: 0
    aten::size_3242:
      type: Shape
      input:
        ret2089.1: {}
      output:
        '6989': {}
      attr:
        start: 1
        end: 1
    aten::mul_730:
      type: Mul
      input:
        ret2089.1: {}
        '892': {}
      output:
        ret2092.1: {}
      attr:
        algorithm: mul
    aten::linear_3611:
      type: InnerProduct
      input:
        ret2092.1: {}
        '893': {}
        aten::linear_3611_bias: {}
      output:
        ret2095.1: {}
    aten::view_5298:
      type: View
      input:
        ret2095.1: {}
        '6987': {}
        '6989': {}
      output:
        ret2096.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2864:
      type: Reorder
      input:
        ret2096.1: {}
      output:
        ret2097.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_732:
      type: Mul
      input:
        ret2089.1: {}
        '894': {}
      output:
        ret2098.1: {}
      attr:
        algorithm: mul
    aten::linear_3612:
      type: InnerProduct
      input:
        ret2098.1: {}
        '895': {}
        aten::linear_3612_bias: {}
      output:
        ret2101.1: {}
    aten::view_5299:
      type: View
      input:
        ret2101.1: {}
        '6987': {}
        '6989': {}
      output:
        ret2102.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2865:
      type: Reorder
      input:
        ret2102.1: {}
      output:
        ret2103.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_734:
      type: Mul
      input:
        ret2089.1: {}
        '896': {}
      output:
        ret2104.1: {}
      attr:
        algorithm: mul
    aten::linear_3613:
      type: InnerProduct
      input:
        ret2104.1: {}
        '897': {}
        aten::linear_3613_bias: {}
      output:
        ret2107.1: {}
    aten::view_5300:
      type: View
      input:
        ret2107.1: {}
        '6987': {}
        '6989': {}
      output:
        ret2108.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2866:
      type: Reorder
      input:
        ret2108.1: {}
      output:
        ret2109.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2867:
      type: Shape
      input:
        ret2103.1: {}
      output:
        '7035': {}
      attr:
        start: 2
        end: 2
    aten::size_2868:
      type: Shape
      input:
        x49.1: {}
      output:
        '7038': {}
      attr:
        start: 2
        end: 2
    aten::add_3372:
      type: Add
      input:
        '7035': {}
        '7038': {}
      output:
        seq_len24.1: {}
    aten::slice_121:
      type: Slice
      input:
        '493': {}
        seq_len24.1: {}
      output:
        '7045': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_161:
      type: Slice
      input:
        '494': {}
        seq_len24.1: {}
      output:
        '7047': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2869:
      type: Shape
      input:
        ret2097.1: {}
      output:
        '7049': {}
      attr:
        start: 2
        end: 2
    aten::add_3243:
      type: Add
      input:
        '7049': {}
        '7038': {}
      output:
        '7054': {}
    aten::slice_2870:
      type: Slice
      input:
        '7045': {}
        '7038': {}
        '7054': {}
      output:
        '7056': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2270:
      type: Slice
      input:
        '7056': {}
      output:
        '7057': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2871:
      type: Slice
      input:
        '7047': {}
        '7038': {}
        '7054': {}
      output:
        '7058': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2271:
      type: Slice
      input:
        '7058': {}
      output:
        '7059': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5301:
      type: Mul
      input:
        ret2097.1: {}
        '7057': {}
      output:
        ret2113.1: {}
      attr:
        algorithm: mul
    aten::size_2272:
      type: Shape
      input:
        ret2097.1: {}
      output:
        '7063': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_226:
      type: Div
      input:
        '7063': {}
        '495': {}
      output:
        ret2115.1: {}
      attr:
        algorithm: div
    aten::slice_2273:
      type: Slice
      input:
        ret2097.1: {}
        ret2115.1: {}
      output:
        ret2116.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2274:
      type: Slice
      input:
        ret2097.1: {}
        ret2115.1: {}
      output:
        ret2117.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5308:
      type: Neg
      input:
        ret2117.1: {}
        aten::neg_5308_mul_val: {}
      output:
        ret2118.1: {}
      attr:
        algorithm: mul
    aten::cat_2503:
      type: Concat
      input:
        ret2118.1: {}
        ret2116.1: {}
      output:
        ret2119.1: {}
      attr:
        axis: -1
    aten::mul_5305:
      type: Mul
      input:
        ret2119.1: {}
        '7059': {}
      output:
        ret2120.1: {}
      attr:
        algorithm: mul
    aten::add_3244:
      type: Add
      input:
        ret2113.1: {}
        ret2120.1: {}
      output:
        args99.1: {}
    aten::mul_5303:
      type: Mul
      input:
        ret2103.1: {}
        '7057': {}
      output:
        ret2121.1: {}
      attr:
        algorithm: mul
    aten::size_2275:
      type: Shape
      input:
        ret2103.1: {}
      output:
        '7085': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_227:
      type: Div
      input:
        '7085': {}
        '495': {}
      output:
        ret2123.1: {}
      attr:
        algorithm: div
    aten::slice_2276:
      type: Slice
      input:
        ret2103.1: {}
        ret2123.1: {}
      output:
        ret2124.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2277:
      type: Slice
      input:
        ret2103.1: {}
        ret2123.1: {}
      output:
        ret2125.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5310:
      type: Neg
      input:
        ret2125.1: {}
        aten::neg_5310_mul_val: {}
      output:
        ret2126.1: {}
      attr:
        algorithm: mul
    aten::cat_2504:
      type: Concat
      input:
        ret2126.1: {}
        ret2124.1: {}
      output:
        ret2127.1: {}
      attr:
        axis: -1
    aten::mul_5306:
      type: Mul
      input:
        ret2127.1: {}
        '7059': {}
      output:
        ret2128.1: {}
      attr:
        algorithm: mul
    aten::add_3245:
      type: Add
      input:
        ret2121.1: {}
        ret2128.1: {}
      output:
        '7103': {}
    aten::cat_2872:
      type: Concat
      input:
        x49.1: {}
        '7103': {}
      output:
        ret2129.1: {}
      attr:
        axis: 2
    aten::cat_2873:
      type: Concat
      input:
        x50.1: {}
        ret2109.1: {}
      output:
        ret2130.1: {}
      attr:
        axis: 2
    aten::transpose_2278:
      type: Reorder
      input:
        ret2129.1: {}
      output:
        ret2131.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5313:
      type: Matmul
      input:
        args99.1: {}
        ret2131.1: {}
      output:
        ret2134.1: {}
    aten::div_281:
      type: Div
      input:
        ret2134.1: {}
        '496': {}
      output:
        ret2135.1: {}
      attr:
        algorithm: div
    aten::add_3246:
      type: Add
      input:
        ret2135.1: {}
        attention_mask0.1: {}
      output:
        attn_weights50.1: {}
    aten::max_321:
      type: Max
      input:
        attn_weights50.1: {}
        '497': {}
      output:
        input50.1: {}
    aten::softmax_2434:
      type: Softmax
      input:
        input50.1: {}
      output:
        '7125': {}
      attr:
        axis: -1
    aten::matmul_5316:
      type: Matmul
      input:
        '7125': {}
        ret2130.1: {}
      output:
        ret2138.1: {}
    aten::transpose_2874:
      type: Reorder
      input:
        ret2138.1: {}
      output:
        ret2139.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5318:
      type: Reshape
      input:
        ret2139.1: {}
        '6987': {}
        '6989': {}
      output:
        ret2140.1: {}
    aten::mul_736:
      type: Mul
      input:
        ret2140.1: {}
        '898': {}
      output:
        ret2141.1: {}
      attr:
        algorithm: mul
    aten::linear_3614:
      type: InnerProduct
      input:
        ret2141.1: {}
        '899': {}
        aten::linear_3614_bias: {}
      output:
        ret2144.1: {}
    aten::add_3247:
      type: Add
      input:
        ret2083.1: {}
        ret2144.1: {}
      output:
        ret2145.1: {}
    aten::pow_2875:
      type: Pow
      input:
        ret2145.1: {}
        aten::pow_2875_other: {}
      output:
        ret2146.1: {}
    aten::mean_1024:
      type: ReduceMean
      input:
        ret2146.1: {}
      output:
        ret2147.1: {}
    aten::add_59:
      type: Add
      input:
        ret2147.1: {}
        '485': {}
      output:
        ret2148.1: {}
    aten::rsqrt_5321:
      type: Rsqrt
      input:
        ret2148.1: {}
      output:
        ret2149.1: {}
    aten::mul_5320:
      type: Mul
      input:
        ret2145.1: {}
        ret2149.1: {}
      output:
        ret2150.1: {}
      attr:
        algorithm: mul
    aten::mul_738:
      type: Mul
      input:
        '900': {}
        ret2150.1: {}
      output:
        ret2151.1: {}
      attr:
        algorithm: mul
    aten::mul_739:
      type: Mul
      input:
        ret2151.1: {}
        '901': {}
      output:
        ret2152.1: {}
      attr:
        algorithm: mul
    aten::linear_3615:
      type: InnerProduct
      input:
        ret2152.1: {}
        '902': {}
        aten::linear_3615_bias: {}
      output:
        ret2155.1: {}
    aten::silu_5323:
      type: Swish
      input:
        ret2155.1: {}
      output:
        ret2156.1: {}
    aten::mul_741:
      type: Mul
      input:
        ret2151.1: {}
        '903': {}
      output:
        ret2157.1: {}
      attr:
        algorithm: mul
    aten::linear_3616:
      type: InnerProduct
      input:
        ret2157.1: {}
        '904': {}
        aten::linear_3616_bias: {}
      output:
        ret2160.1: {}
    aten::mul_5324:
      type: Mul
      input:
        ret2156.1: {}
        ret2160.1: {}
      output:
        ret2161.1: {}
      attr:
        algorithm: mul
    aten::mul_743:
      type: Mul
      input:
        ret2161.1: {}
        '905': {}
      output:
        ret2162.1: {}
      attr:
        algorithm: mul
    aten::linear_3617:
      type: InnerProduct
      input:
        ret2162.1: {}
        '906': {}
        aten::linear_3617_bias: {}
      output:
        ret2165.1: {}
    aten::add_3248:
      type: Add
      input:
        ret2145.1: {}
        ret2165.1: {}
      output:
        ret2166.1: {}
    aten::pow_2876:
      type: Pow
      input:
        ret2166.1: {}
        aten::pow_2876_other: {}
      output:
        ret2167.1: {}
    aten::mean_1025:
      type: ReduceMean
      input:
        ret2167.1: {}
      output:
        ret2168.1: {}
    aten::add_60:
      type: Add
      input:
        ret2168.1: {}
        '485': {}
      output:
        ret2169.1: {}
    aten::rsqrt_5328:
      type: Rsqrt
      input:
        ret2169.1: {}
      output:
        ret2170.1: {}
    aten::mul_5327:
      type: Mul
      input:
        ret2166.1: {}
        ret2170.1: {}
      output:
        ret2171.1: {}
      attr:
        algorithm: mul
    aten::mul_745:
      type: Mul
      input:
        '907': {}
        ret2171.1: {}
      output:
        ret2172.1: {}
      attr:
        algorithm: mul
    aten::size_3415:
      type: Shape
      input:
        ret2172.1: {}
      output:
        '7210': {}
      attr:
        start: 0
        end: 0
    aten::size_3249:
      type: Shape
      input:
        ret2172.1: {}
      output:
        '7212': {}
      attr:
        start: 1
        end: 1
    aten::mul_746:
      type: Mul
      input:
        ret2172.1: {}
        '908': {}
      output:
        ret2175.1: {}
      attr:
        algorithm: mul
    aten::linear_3618:
      type: InnerProduct
      input:
        ret2175.1: {}
        '909': {}
        aten::linear_3618_bias: {}
      output:
        ret2178.1: {}
    aten::view_5330:
      type: View
      input:
        ret2178.1: {}
        '7210': {}
        '7212': {}
      output:
        ret2179.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2877:
      type: Reorder
      input:
        ret2179.1: {}
      output:
        ret2180.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_748:
      type: Mul
      input:
        ret2172.1: {}
        '910': {}
      output:
        ret2181.1: {}
      attr:
        algorithm: mul
    aten::linear_3619:
      type: InnerProduct
      input:
        ret2181.1: {}
        '911': {}
        aten::linear_3619_bias: {}
      output:
        ret2184.1: {}
    aten::view_5331:
      type: View
      input:
        ret2184.1: {}
        '7210': {}
        '7212': {}
      output:
        ret2185.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2878:
      type: Reorder
      input:
        ret2185.1: {}
      output:
        ret2186.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_750:
      type: Mul
      input:
        ret2172.1: {}
        '912': {}
      output:
        ret2187.1: {}
      attr:
        algorithm: mul
    aten::linear_3620:
      type: InnerProduct
      input:
        ret2187.1: {}
        '913': {}
        aten::linear_3620_bias: {}
      output:
        ret2190.1: {}
    aten::view_5332:
      type: View
      input:
        ret2190.1: {}
        '7210': {}
        '7212': {}
      output:
        ret2191.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2879:
      type: Reorder
      input:
        ret2191.1: {}
      output:
        ret2192.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2880:
      type: Shape
      input:
        ret2186.1: {}
      output:
        '7258': {}
      attr:
        start: 2
        end: 2
    aten::size_2881:
      type: Shape
      input:
        x51.1: {}
      output:
        '7261': {}
      attr:
        start: 2
        end: 2
    aten::add_3373:
      type: Add
      input:
        '7258': {}
        '7261': {}
      output:
        seq_len25.1: {}
    aten::slice_122:
      type: Slice
      input:
        '493': {}
        seq_len25.1: {}
      output:
        '7268': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_162:
      type: Slice
      input:
        '494': {}
        seq_len25.1: {}
      output:
        '7270': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2882:
      type: Shape
      input:
        ret2180.1: {}
      output:
        '7272': {}
      attr:
        start: 2
        end: 2
    aten::add_3250:
      type: Add
      input:
        '7272': {}
        '7261': {}
      output:
        '7277': {}
    aten::slice_2883:
      type: Slice
      input:
        '7268': {}
        '7261': {}
        '7277': {}
      output:
        '7279': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2279:
      type: Slice
      input:
        '7279': {}
      output:
        '7280': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2884:
      type: Slice
      input:
        '7270': {}
        '7261': {}
        '7277': {}
      output:
        '7281': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2280:
      type: Slice
      input:
        '7281': {}
      output:
        '7282': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5333:
      type: Mul
      input:
        ret2180.1: {}
        '7280': {}
      output:
        ret2196.1: {}
      attr:
        algorithm: mul
    aten::size_2281:
      type: Shape
      input:
        ret2180.1: {}
      output:
        '7286': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_228:
      type: Div
      input:
        '7286': {}
        '495': {}
      output:
        ret2198.1: {}
      attr:
        algorithm: div
    aten::slice_2282:
      type: Slice
      input:
        ret2180.1: {}
        ret2198.1: {}
      output:
        ret2199.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2283:
      type: Slice
      input:
        ret2180.1: {}
        ret2198.1: {}
      output:
        ret2200.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5340:
      type: Neg
      input:
        ret2200.1: {}
        aten::neg_5340_mul_val: {}
      output:
        ret2201.1: {}
      attr:
        algorithm: mul
    aten::cat_2505:
      type: Concat
      input:
        ret2201.1: {}
        ret2199.1: {}
      output:
        ret2202.1: {}
      attr:
        axis: -1
    aten::mul_5337:
      type: Mul
      input:
        ret2202.1: {}
        '7282': {}
      output:
        ret2203.1: {}
      attr:
        algorithm: mul
    aten::add_3251:
      type: Add
      input:
        ret2196.1: {}
        ret2203.1: {}
      output:
        args103.1: {}
    aten::mul_5335:
      type: Mul
      input:
        ret2186.1: {}
        '7280': {}
      output:
        ret2204.1: {}
      attr:
        algorithm: mul
    aten::size_2284:
      type: Shape
      input:
        ret2186.1: {}
      output:
        '7308': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_229:
      type: Div
      input:
        '7308': {}
        '495': {}
      output:
        ret2206.1: {}
      attr:
        algorithm: div
    aten::slice_2285:
      type: Slice
      input:
        ret2186.1: {}
        ret2206.1: {}
      output:
        ret2207.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2286:
      type: Slice
      input:
        ret2186.1: {}
        ret2206.1: {}
      output:
        ret2208.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5342:
      type: Neg
      input:
        ret2208.1: {}
        aten::neg_5342_mul_val: {}
      output:
        ret2209.1: {}
      attr:
        algorithm: mul
    aten::cat_2506:
      type: Concat
      input:
        ret2209.1: {}
        ret2207.1: {}
      output:
        ret2210.1: {}
      attr:
        axis: -1
    aten::mul_5338:
      type: Mul
      input:
        ret2210.1: {}
        '7282': {}
      output:
        ret2211.1: {}
      attr:
        algorithm: mul
    aten::add_3252:
      type: Add
      input:
        ret2204.1: {}
        ret2211.1: {}
      output:
        '7326': {}
    aten::cat_2885:
      type: Concat
      input:
        x51.1: {}
        '7326': {}
      output:
        ret2212.1: {}
      attr:
        axis: 2
    aten::cat_2886:
      type: Concat
      input:
        x52.1: {}
        ret2192.1: {}
      output:
        ret2213.1: {}
      attr:
        axis: 2
    aten::transpose_2287:
      type: Reorder
      input:
        ret2212.1: {}
      output:
        ret2214.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5345:
      type: Matmul
      input:
        args103.1: {}
        ret2214.1: {}
      output:
        ret2217.1: {}
    aten::div_282:
      type: Div
      input:
        ret2217.1: {}
        '496': {}
      output:
        ret2218.1: {}
      attr:
        algorithm: div
    aten::add_3253:
      type: Add
      input:
        ret2218.1: {}
        attention_mask0.1: {}
      output:
        attn_weights52.1: {}
    aten::max_322:
      type: Max
      input:
        attn_weights52.1: {}
        '497': {}
      output:
        input52.1: {}
    aten::softmax_2435:
      type: Softmax
      input:
        input52.1: {}
      output:
        '7348': {}
      attr:
        axis: -1
    aten::matmul_5348:
      type: Matmul
      input:
        '7348': {}
        ret2213.1: {}
      output:
        ret2221.1: {}
    aten::transpose_2887:
      type: Reorder
      input:
        ret2221.1: {}
      output:
        ret2222.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5350:
      type: Reshape
      input:
        ret2222.1: {}
        '7210': {}
        '7212': {}
      output:
        ret2223.1: {}
    aten::mul_752:
      type: Mul
      input:
        ret2223.1: {}
        '914': {}
      output:
        ret2224.1: {}
      attr:
        algorithm: mul
    aten::linear_3621:
      type: InnerProduct
      input:
        ret2224.1: {}
        '915': {}
        aten::linear_3621_bias: {}
      output:
        ret2227.1: {}
    aten::add_3254:
      type: Add
      input:
        ret2166.1: {}
        ret2227.1: {}
      output:
        ret2228.1: {}
    aten::pow_2888:
      type: Pow
      input:
        ret2228.1: {}
        aten::pow_2888_other: {}
      output:
        ret2229.1: {}
    aten::mean_1026:
      type: ReduceMean
      input:
        ret2229.1: {}
      output:
        ret2230.1: {}
    aten::add_61:
      type: Add
      input:
        ret2230.1: {}
        '485': {}
      output:
        ret2231.1: {}
    aten::rsqrt_5353:
      type: Rsqrt
      input:
        ret2231.1: {}
      output:
        ret2232.1: {}
    aten::mul_5352:
      type: Mul
      input:
        ret2228.1: {}
        ret2232.1: {}
      output:
        ret2233.1: {}
      attr:
        algorithm: mul
    aten::mul_754:
      type: Mul
      input:
        '916': {}
        ret2233.1: {}
      output:
        ret2234.1: {}
      attr:
        algorithm: mul
    aten::mul_755:
      type: Mul
      input:
        ret2234.1: {}
        '917': {}
      output:
        ret2235.1: {}
      attr:
        algorithm: mul
    aten::linear_3622:
      type: InnerProduct
      input:
        ret2235.1: {}
        '918': {}
        aten::linear_3622_bias: {}
      output:
        ret2238.1: {}
    aten::silu_5355:
      type: Swish
      input:
        ret2238.1: {}
      output:
        ret2239.1: {}
    aten::mul_757:
      type: Mul
      input:
        ret2234.1: {}
        '919': {}
      output:
        ret2240.1: {}
      attr:
        algorithm: mul
    aten::linear_3623:
      type: InnerProduct
      input:
        ret2240.1: {}
        '920': {}
        aten::linear_3623_bias: {}
      output:
        ret2243.1: {}
    aten::mul_5356:
      type: Mul
      input:
        ret2239.1: {}
        ret2243.1: {}
      output:
        ret2244.1: {}
      attr:
        algorithm: mul
    aten::mul_759:
      type: Mul
      input:
        ret2244.1: {}
        '921': {}
      output:
        ret2245.1: {}
      attr:
        algorithm: mul
    aten::linear_3624:
      type: InnerProduct
      input:
        ret2245.1: {}
        '922': {}
        aten::linear_3624_bias: {}
      output:
        ret2248.1: {}
    aten::add_3255:
      type: Add
      input:
        ret2228.1: {}
        ret2248.1: {}
      output:
        ret2249.1: {}
    aten::pow_2889:
      type: Pow
      input:
        ret2249.1: {}
        aten::pow_2889_other: {}
      output:
        ret2250.1: {}
    aten::mean_1027:
      type: ReduceMean
      input:
        ret2250.1: {}
      output:
        ret2251.1: {}
    aten::add_62:
      type: Add
      input:
        ret2251.1: {}
        '485': {}
      output:
        ret2252.1: {}
    aten::rsqrt_5360:
      type: Rsqrt
      input:
        ret2252.1: {}
      output:
        ret2253.1: {}
    aten::mul_5359:
      type: Mul
      input:
        ret2249.1: {}
        ret2253.1: {}
      output:
        ret2254.1: {}
      attr:
        algorithm: mul
    aten::mul_761:
      type: Mul
      input:
        '923': {}
        ret2254.1: {}
      output:
        ret2255.1: {}
      attr:
        algorithm: mul
    aten::size_3416:
      type: Shape
      input:
        ret2255.1: {}
      output:
        '7433': {}
      attr:
        start: 0
        end: 0
    aten::size_3256:
      type: Shape
      input:
        ret2255.1: {}
      output:
        '7435': {}
      attr:
        start: 1
        end: 1
    aten::mul_762:
      type: Mul
      input:
        ret2255.1: {}
        '924': {}
      output:
        ret2258.1: {}
      attr:
        algorithm: mul
    aten::linear_3625:
      type: InnerProduct
      input:
        ret2258.1: {}
        '925': {}
        aten::linear_3625_bias: {}
      output:
        ret2261.1: {}
    aten::view_5362:
      type: View
      input:
        ret2261.1: {}
        '7433': {}
        '7435': {}
      output:
        ret2262.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2890:
      type: Reorder
      input:
        ret2262.1: {}
      output:
        ret2263.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_764:
      type: Mul
      input:
        ret2255.1: {}
        '926': {}
      output:
        ret2264.1: {}
      attr:
        algorithm: mul
    aten::linear_3626:
      type: InnerProduct
      input:
        ret2264.1: {}
        '927': {}
        aten::linear_3626_bias: {}
      output:
        ret2267.1: {}
    aten::view_5363:
      type: View
      input:
        ret2267.1: {}
        '7433': {}
        '7435': {}
      output:
        ret2268.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2891:
      type: Reorder
      input:
        ret2268.1: {}
      output:
        ret2269.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_766:
      type: Mul
      input:
        ret2255.1: {}
        '928': {}
      output:
        ret2270.1: {}
      attr:
        algorithm: mul
    aten::linear_3627:
      type: InnerProduct
      input:
        ret2270.1: {}
        '929': {}
        aten::linear_3627_bias: {}
      output:
        ret2273.1: {}
    aten::view_5364:
      type: View
      input:
        ret2273.1: {}
        '7433': {}
        '7435': {}
      output:
        ret2274.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2892:
      type: Reorder
      input:
        ret2274.1: {}
      output:
        ret2275.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2893:
      type: Shape
      input:
        ret2269.1: {}
      output:
        '7481': {}
      attr:
        start: 2
        end: 2
    aten::size_2894:
      type: Shape
      input:
        x53.1: {}
      output:
        '7484': {}
      attr:
        start: 2
        end: 2
    aten::add_3374:
      type: Add
      input:
        '7481': {}
        '7484': {}
      output:
        seq_len26.1: {}
    aten::slice_123:
      type: Slice
      input:
        '493': {}
        seq_len26.1: {}
      output:
        '7491': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_163:
      type: Slice
      input:
        '494': {}
        seq_len26.1: {}
      output:
        '7493': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2895:
      type: Shape
      input:
        ret2263.1: {}
      output:
        '7495': {}
      attr:
        start: 2
        end: 2
    aten::add_3257:
      type: Add
      input:
        '7495': {}
        '7484': {}
      output:
        '7500': {}
    aten::slice_2896:
      type: Slice
      input:
        '7491': {}
        '7484': {}
        '7500': {}
      output:
        '7502': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2288:
      type: Slice
      input:
        '7502': {}
      output:
        '7503': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2897:
      type: Slice
      input:
        '7493': {}
        '7484': {}
        '7500': {}
      output:
        '7504': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2289:
      type: Slice
      input:
        '7504': {}
      output:
        '7505': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5365:
      type: Mul
      input:
        ret2263.1: {}
        '7503': {}
      output:
        ret2279.1: {}
      attr:
        algorithm: mul
    aten::size_2290:
      type: Shape
      input:
        ret2263.1: {}
      output:
        '7509': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_230:
      type: Div
      input:
        '7509': {}
        '495': {}
      output:
        ret2281.1: {}
      attr:
        algorithm: div
    aten::slice_2291:
      type: Slice
      input:
        ret2263.1: {}
        ret2281.1: {}
      output:
        ret2282.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2292:
      type: Slice
      input:
        ret2263.1: {}
        ret2281.1: {}
      output:
        ret2283.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5372:
      type: Neg
      input:
        ret2283.1: {}
        aten::neg_5372_mul_val: {}
      output:
        ret2284.1: {}
      attr:
        algorithm: mul
    aten::cat_2507:
      type: Concat
      input:
        ret2284.1: {}
        ret2282.1: {}
      output:
        ret2285.1: {}
      attr:
        axis: -1
    aten::mul_5369:
      type: Mul
      input:
        ret2285.1: {}
        '7505': {}
      output:
        ret2286.1: {}
      attr:
        algorithm: mul
    aten::add_3258:
      type: Add
      input:
        ret2279.1: {}
        ret2286.1: {}
      output:
        args107.1: {}
    aten::mul_5367:
      type: Mul
      input:
        ret2269.1: {}
        '7503': {}
      output:
        ret2287.1: {}
      attr:
        algorithm: mul
    aten::size_2293:
      type: Shape
      input:
        ret2269.1: {}
      output:
        '7531': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_231:
      type: Div
      input:
        '7531': {}
        '495': {}
      output:
        ret2289.1: {}
      attr:
        algorithm: div
    aten::slice_2294:
      type: Slice
      input:
        ret2269.1: {}
        ret2289.1: {}
      output:
        ret2290.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2295:
      type: Slice
      input:
        ret2269.1: {}
        ret2289.1: {}
      output:
        ret2291.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5374:
      type: Neg
      input:
        ret2291.1: {}
        aten::neg_5374_mul_val: {}
      output:
        ret2292.1: {}
      attr:
        algorithm: mul
    aten::cat_2508:
      type: Concat
      input:
        ret2292.1: {}
        ret2290.1: {}
      output:
        ret2293.1: {}
      attr:
        axis: -1
    aten::mul_5370:
      type: Mul
      input:
        ret2293.1: {}
        '7505': {}
      output:
        ret2294.1: {}
      attr:
        algorithm: mul
    aten::add_3259:
      type: Add
      input:
        ret2287.1: {}
        ret2294.1: {}
      output:
        '7549': {}
    aten::cat_2898:
      type: Concat
      input:
        x53.1: {}
        '7549': {}
      output:
        ret2295.1: {}
      attr:
        axis: 2
    aten::cat_2899:
      type: Concat
      input:
        x54.1: {}
        ret2275.1: {}
      output:
        ret2296.1: {}
      attr:
        axis: 2
    aten::transpose_2296:
      type: Reorder
      input:
        ret2295.1: {}
      output:
        ret2297.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5377:
      type: Matmul
      input:
        args107.1: {}
        ret2297.1: {}
      output:
        ret2300.1: {}
    aten::div_283:
      type: Div
      input:
        ret2300.1: {}
        '496': {}
      output:
        ret2301.1: {}
      attr:
        algorithm: div
    aten::add_3260:
      type: Add
      input:
        ret2301.1: {}
        attention_mask0.1: {}
      output:
        attn_weights54.1: {}
    aten::max_323:
      type: Max
      input:
        attn_weights54.1: {}
        '497': {}
      output:
        input54.1: {}
    aten::softmax_2436:
      type: Softmax
      input:
        input54.1: {}
      output:
        '7571': {}
      attr:
        axis: -1
    aten::matmul_5380:
      type: Matmul
      input:
        '7571': {}
        ret2296.1: {}
      output:
        ret2304.1: {}
    aten::transpose_2900:
      type: Reorder
      input:
        ret2304.1: {}
      output:
        ret2305.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5382:
      type: Reshape
      input:
        ret2305.1: {}
        '7433': {}
        '7435': {}
      output:
        ret2306.1: {}
    aten::mul_768:
      type: Mul
      input:
        ret2306.1: {}
        '930': {}
      output:
        ret2307.1: {}
      attr:
        algorithm: mul
    aten::linear_3628:
      type: InnerProduct
      input:
        ret2307.1: {}
        '931': {}
        aten::linear_3628_bias: {}
      output:
        ret2310.1: {}
    aten::add_3261:
      type: Add
      input:
        ret2249.1: {}
        ret2310.1: {}
      output:
        ret2311.1: {}
    aten::pow_2901:
      type: Pow
      input:
        ret2311.1: {}
        aten::pow_2901_other: {}
      output:
        ret2312.1: {}
    aten::mean_1028:
      type: ReduceMean
      input:
        ret2312.1: {}
      output:
        ret2313.1: {}
    aten::add_63:
      type: Add
      input:
        ret2313.1: {}
        '485': {}
      output:
        ret2314.1: {}
    aten::rsqrt_5385:
      type: Rsqrt
      input:
        ret2314.1: {}
      output:
        ret2315.1: {}
    aten::mul_5384:
      type: Mul
      input:
        ret2311.1: {}
        ret2315.1: {}
      output:
        ret2316.1: {}
      attr:
        algorithm: mul
    aten::mul_770:
      type: Mul
      input:
        '932': {}
        ret2316.1: {}
      output:
        ret2317.1: {}
      attr:
        algorithm: mul
    aten::mul_771:
      type: Mul
      input:
        ret2317.1: {}
        '933': {}
      output:
        ret2318.1: {}
      attr:
        algorithm: mul
    aten::linear_3629:
      type: InnerProduct
      input:
        ret2318.1: {}
        '934': {}
        aten::linear_3629_bias: {}
      output:
        ret2321.1: {}
    aten::silu_5387:
      type: Swish
      input:
        ret2321.1: {}
      output:
        ret2322.1: {}
    aten::mul_773:
      type: Mul
      input:
        ret2317.1: {}
        '935': {}
      output:
        ret2323.1: {}
      attr:
        algorithm: mul
    aten::linear_3630:
      type: InnerProduct
      input:
        ret2323.1: {}
        '936': {}
        aten::linear_3630_bias: {}
      output:
        ret2326.1: {}
    aten::mul_5388:
      type: Mul
      input:
        ret2322.1: {}
        ret2326.1: {}
      output:
        ret2327.1: {}
      attr:
        algorithm: mul
    aten::mul_775:
      type: Mul
      input:
        ret2327.1: {}
        '937': {}
      output:
        ret2328.1: {}
      attr:
        algorithm: mul
    aten::linear_3631:
      type: InnerProduct
      input:
        ret2328.1: {}
        '938': {}
        aten::linear_3631_bias: {}
      output:
        ret2331.1: {}
    aten::add_3262:
      type: Add
      input:
        ret2311.1: {}
        ret2331.1: {}
      output:
        ret2332.1: {}
    aten::pow_2902:
      type: Pow
      input:
        ret2332.1: {}
        aten::pow_2902_other: {}
      output:
        ret2333.1: {}
    aten::mean_1029:
      type: ReduceMean
      input:
        ret2333.1: {}
      output:
        ret2334.1: {}
    aten::add_64:
      type: Add
      input:
        ret2334.1: {}
        '485': {}
      output:
        ret2335.1: {}
    aten::rsqrt_5392:
      type: Rsqrt
      input:
        ret2335.1: {}
      output:
        ret2336.1: {}
    aten::mul_5391:
      type: Mul
      input:
        ret2332.1: {}
        ret2336.1: {}
      output:
        ret2337.1: {}
      attr:
        algorithm: mul
    aten::mul_777:
      type: Mul
      input:
        '939': {}
        ret2337.1: {}
      output:
        ret2338.1: {}
      attr:
        algorithm: mul
    aten::size_3417:
      type: Shape
      input:
        ret2338.1: {}
      output:
        '7656': {}
      attr:
        start: 0
        end: 0
    aten::size_3263:
      type: Shape
      input:
        ret2338.1: {}
      output:
        '7658': {}
      attr:
        start: 1
        end: 1
    aten::mul_778:
      type: Mul
      input:
        ret2338.1: {}
        '940': {}
      output:
        ret2341.1: {}
      attr:
        algorithm: mul
    aten::linear_3632:
      type: InnerProduct
      input:
        ret2341.1: {}
        '941': {}
        aten::linear_3632_bias: {}
      output:
        ret2344.1: {}
    aten::view_5394:
      type: View
      input:
        ret2344.1: {}
        '7656': {}
        '7658': {}
      output:
        ret2345.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2903:
      type: Reorder
      input:
        ret2345.1: {}
      output:
        ret2346.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_780:
      type: Mul
      input:
        ret2338.1: {}
        '942': {}
      output:
        ret2347.1: {}
      attr:
        algorithm: mul
    aten::linear_3633:
      type: InnerProduct
      input:
        ret2347.1: {}
        '943': {}
        aten::linear_3633_bias: {}
      output:
        ret2350.1: {}
    aten::view_5395:
      type: View
      input:
        ret2350.1: {}
        '7656': {}
        '7658': {}
      output:
        ret2351.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2904:
      type: Reorder
      input:
        ret2351.1: {}
      output:
        ret2352.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_782:
      type: Mul
      input:
        ret2338.1: {}
        '944': {}
      output:
        ret2353.1: {}
      attr:
        algorithm: mul
    aten::linear_3634:
      type: InnerProduct
      input:
        ret2353.1: {}
        '945': {}
        aten::linear_3634_bias: {}
      output:
        ret2356.1: {}
    aten::view_5396:
      type: View
      input:
        ret2356.1: {}
        '7656': {}
        '7658': {}
      output:
        ret2357.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2905:
      type: Reorder
      input:
        ret2357.1: {}
      output:
        ret2358.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2906:
      type: Shape
      input:
        ret2352.1: {}
      output:
        '7704': {}
      attr:
        start: 2
        end: 2
    aten::size_2907:
      type: Shape
      input:
        x55.1: {}
      output:
        '7707': {}
      attr:
        start: 2
        end: 2
    aten::add_3375:
      type: Add
      input:
        '7704': {}
        '7707': {}
      output:
        seq_len27.1: {}
    aten::slice_124:
      type: Slice
      input:
        '493': {}
        seq_len27.1: {}
      output:
        '7714': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_164:
      type: Slice
      input:
        '494': {}
        seq_len27.1: {}
      output:
        '7716': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2908:
      type: Shape
      input:
        ret2346.1: {}
      output:
        '7718': {}
      attr:
        start: 2
        end: 2
    aten::add_3264:
      type: Add
      input:
        '7718': {}
        '7707': {}
      output:
        '7723': {}
    aten::slice_2909:
      type: Slice
      input:
        '7714': {}
        '7707': {}
        '7723': {}
      output:
        '7725': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2297:
      type: Slice
      input:
        '7725': {}
      output:
        '7726': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2910:
      type: Slice
      input:
        '7716': {}
        '7707': {}
        '7723': {}
      output:
        '7727': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2298:
      type: Slice
      input:
        '7727': {}
      output:
        '7728': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5397:
      type: Mul
      input:
        ret2346.1: {}
        '7726': {}
      output:
        ret2362.1: {}
      attr:
        algorithm: mul
    aten::size_2299:
      type: Shape
      input:
        ret2346.1: {}
      output:
        '7732': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_232:
      type: Div
      input:
        '7732': {}
        '495': {}
      output:
        ret2364.1: {}
      attr:
        algorithm: div
    aten::slice_2300:
      type: Slice
      input:
        ret2346.1: {}
        ret2364.1: {}
      output:
        ret2365.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2301:
      type: Slice
      input:
        ret2346.1: {}
        ret2364.1: {}
      output:
        ret2366.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5404:
      type: Neg
      input:
        ret2366.1: {}
        aten::neg_5404_mul_val: {}
      output:
        ret2367.1: {}
      attr:
        algorithm: mul
    aten::cat_2509:
      type: Concat
      input:
        ret2367.1: {}
        ret2365.1: {}
      output:
        ret2368.1: {}
      attr:
        axis: -1
    aten::mul_5401:
      type: Mul
      input:
        ret2368.1: {}
        '7728': {}
      output:
        ret2369.1: {}
      attr:
        algorithm: mul
    aten::add_3265:
      type: Add
      input:
        ret2362.1: {}
        ret2369.1: {}
      output:
        args111.1: {}
    aten::mul_5399:
      type: Mul
      input:
        ret2352.1: {}
        '7726': {}
      output:
        ret2370.1: {}
      attr:
        algorithm: mul
    aten::size_2302:
      type: Shape
      input:
        ret2352.1: {}
      output:
        '7754': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_233:
      type: Div
      input:
        '7754': {}
        '495': {}
      output:
        ret2372.1: {}
      attr:
        algorithm: div
    aten::slice_2303:
      type: Slice
      input:
        ret2352.1: {}
        ret2372.1: {}
      output:
        ret2373.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2304:
      type: Slice
      input:
        ret2352.1: {}
        ret2372.1: {}
      output:
        ret2374.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5406:
      type: Neg
      input:
        ret2374.1: {}
        aten::neg_5406_mul_val: {}
      output:
        ret2375.1: {}
      attr:
        algorithm: mul
    aten::cat_2510:
      type: Concat
      input:
        ret2375.1: {}
        ret2373.1: {}
      output:
        ret2376.1: {}
      attr:
        axis: -1
    aten::mul_5402:
      type: Mul
      input:
        ret2376.1: {}
        '7728': {}
      output:
        ret2377.1: {}
      attr:
        algorithm: mul
    aten::add_3266:
      type: Add
      input:
        ret2370.1: {}
        ret2377.1: {}
      output:
        '7772': {}
    aten::cat_2911:
      type: Concat
      input:
        x55.1: {}
        '7772': {}
      output:
        ret2378.1: {}
      attr:
        axis: 2
    aten::cat_2912:
      type: Concat
      input:
        x56.1: {}
        ret2358.1: {}
      output:
        ret2379.1: {}
      attr:
        axis: 2
    aten::transpose_2305:
      type: Reorder
      input:
        ret2378.1: {}
      output:
        ret2380.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5409:
      type: Matmul
      input:
        args111.1: {}
        ret2380.1: {}
      output:
        ret2383.1: {}
    aten::div_284:
      type: Div
      input:
        ret2383.1: {}
        '496': {}
      output:
        ret2384.1: {}
      attr:
        algorithm: div
    aten::add_3267:
      type: Add
      input:
        ret2384.1: {}
        attention_mask0.1: {}
      output:
        attn_weights56.1: {}
    aten::max_324:
      type: Max
      input:
        attn_weights56.1: {}
        '497': {}
      output:
        input56.1: {}
    aten::softmax_2437:
      type: Softmax
      input:
        input56.1: {}
      output:
        '7794': {}
      attr:
        axis: -1
    aten::matmul_5412:
      type: Matmul
      input:
        '7794': {}
        ret2379.1: {}
      output:
        ret2387.1: {}
    aten::transpose_2913:
      type: Reorder
      input:
        ret2387.1: {}
      output:
        ret2388.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5414:
      type: Reshape
      input:
        ret2388.1: {}
        '7656': {}
        '7658': {}
      output:
        ret2389.1: {}
    aten::mul_784:
      type: Mul
      input:
        ret2389.1: {}
        '946': {}
      output:
        ret2390.1: {}
      attr:
        algorithm: mul
    aten::linear_3635:
      type: InnerProduct
      input:
        ret2390.1: {}
        '947': {}
        aten::linear_3635_bias: {}
      output:
        ret2393.1: {}
    aten::add_3268:
      type: Add
      input:
        ret2332.1: {}
        ret2393.1: {}
      output:
        ret2394.1: {}
    aten::pow_2914:
      type: Pow
      input:
        ret2394.1: {}
        aten::pow_2914_other: {}
      output:
        ret2395.1: {}
    aten::mean_1030:
      type: ReduceMean
      input:
        ret2395.1: {}
      output:
        ret2396.1: {}
    aten::add_65:
      type: Add
      input:
        ret2396.1: {}
        '485': {}
      output:
        ret2397.1: {}
    aten::rsqrt_5417:
      type: Rsqrt
      input:
        ret2397.1: {}
      output:
        ret2398.1: {}
    aten::mul_5416:
      type: Mul
      input:
        ret2394.1: {}
        ret2398.1: {}
      output:
        ret2399.1: {}
      attr:
        algorithm: mul
    aten::mul_786:
      type: Mul
      input:
        '948': {}
        ret2399.1: {}
      output:
        ret2400.1: {}
      attr:
        algorithm: mul
    aten::mul_787:
      type: Mul
      input:
        ret2400.1: {}
        '949': {}
      output:
        ret2401.1: {}
      attr:
        algorithm: mul
    aten::linear_3636:
      type: InnerProduct
      input:
        ret2401.1: {}
        '950': {}
        aten::linear_3636_bias: {}
      output:
        ret2404.1: {}
    aten::silu_5419:
      type: Swish
      input:
        ret2404.1: {}
      output:
        ret2405.1: {}
    aten::mul_789:
      type: Mul
      input:
        ret2400.1: {}
        '951': {}
      output:
        ret2406.1: {}
      attr:
        algorithm: mul
    aten::linear_3637:
      type: InnerProduct
      input:
        ret2406.1: {}
        '952': {}
        aten::linear_3637_bias: {}
      output:
        ret2409.1: {}
    aten::mul_5420:
      type: Mul
      input:
        ret2405.1: {}
        ret2409.1: {}
      output:
        ret2410.1: {}
      attr:
        algorithm: mul
    aten::mul_791:
      type: Mul
      input:
        ret2410.1: {}
        '953': {}
      output:
        ret2411.1: {}
      attr:
        algorithm: mul
    aten::linear_3638:
      type: InnerProduct
      input:
        ret2411.1: {}
        '954': {}
        aten::linear_3638_bias: {}
      output:
        ret2414.1: {}
    aten::add_3269:
      type: Add
      input:
        ret2394.1: {}
        ret2414.1: {}
      output:
        ret2415.1: {}
    aten::pow_2915:
      type: Pow
      input:
        ret2415.1: {}
        aten::pow_2915_other: {}
      output:
        ret2416.1: {}
    aten::mean_1031:
      type: ReduceMean
      input:
        ret2416.1: {}
      output:
        ret2417.1: {}
    aten::add_66:
      type: Add
      input:
        ret2417.1: {}
        '485': {}
      output:
        ret2418.1: {}
    aten::rsqrt_5424:
      type: Rsqrt
      input:
        ret2418.1: {}
      output:
        ret2419.1: {}
    aten::mul_5423:
      type: Mul
      input:
        ret2415.1: {}
        ret2419.1: {}
      output:
        ret2420.1: {}
      attr:
        algorithm: mul
    aten::mul_793:
      type: Mul
      input:
        '955': {}
        ret2420.1: {}
      output:
        ret2421.1: {}
      attr:
        algorithm: mul
    aten::size_3418:
      type: Shape
      input:
        ret2421.1: {}
      output:
        '7879': {}
      attr:
        start: 0
        end: 0
    aten::size_3270:
      type: Shape
      input:
        ret2421.1: {}
      output:
        '7881': {}
      attr:
        start: 1
        end: 1
    aten::mul_794:
      type: Mul
      input:
        ret2421.1: {}
        '956': {}
      output:
        ret2424.1: {}
      attr:
        algorithm: mul
    aten::linear_3639:
      type: InnerProduct
      input:
        ret2424.1: {}
        '957': {}
        aten::linear_3639_bias: {}
      output:
        ret2427.1: {}
    aten::view_5426:
      type: View
      input:
        ret2427.1: {}
        '7879': {}
        '7881': {}
      output:
        ret2428.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2916:
      type: Reorder
      input:
        ret2428.1: {}
      output:
        ret2429.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_796:
      type: Mul
      input:
        ret2421.1: {}
        '958': {}
      output:
        ret2430.1: {}
      attr:
        algorithm: mul
    aten::linear_3640:
      type: InnerProduct
      input:
        ret2430.1: {}
        '959': {}
        aten::linear_3640_bias: {}
      output:
        ret2433.1: {}
    aten::view_5427:
      type: View
      input:
        ret2433.1: {}
        '7879': {}
        '7881': {}
      output:
        ret2434.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2917:
      type: Reorder
      input:
        ret2434.1: {}
      output:
        ret2435.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_798:
      type: Mul
      input:
        ret2421.1: {}
        '960': {}
      output:
        ret2436.1: {}
      attr:
        algorithm: mul
    aten::linear_3641:
      type: InnerProduct
      input:
        ret2436.1: {}
        '961': {}
        aten::linear_3641_bias: {}
      output:
        ret2439.1: {}
    aten::view_5428:
      type: View
      input:
        ret2439.1: {}
        '7879': {}
        '7881': {}
      output:
        ret2440.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2918:
      type: Reorder
      input:
        ret2440.1: {}
      output:
        ret2441.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2919:
      type: Shape
      input:
        ret2435.1: {}
      output:
        '7927': {}
      attr:
        start: 2
        end: 2
    aten::size_2920:
      type: Shape
      input:
        x57.1: {}
      output:
        '7930': {}
      attr:
        start: 2
        end: 2
    aten::add_3376:
      type: Add
      input:
        '7927': {}
        '7930': {}
      output:
        seq_len28.1: {}
    aten::slice_125:
      type: Slice
      input:
        '493': {}
        seq_len28.1: {}
      output:
        '7937': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_165:
      type: Slice
      input:
        '494': {}
        seq_len28.1: {}
      output:
        '7939': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2921:
      type: Shape
      input:
        ret2429.1: {}
      output:
        '7941': {}
      attr:
        start: 2
        end: 2
    aten::add_3271:
      type: Add
      input:
        '7941': {}
        '7930': {}
      output:
        '7946': {}
    aten::slice_2922:
      type: Slice
      input:
        '7937': {}
        '7930': {}
        '7946': {}
      output:
        '7948': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2306:
      type: Slice
      input:
        '7948': {}
      output:
        '7949': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2923:
      type: Slice
      input:
        '7939': {}
        '7930': {}
        '7946': {}
      output:
        '7950': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2307:
      type: Slice
      input:
        '7950': {}
      output:
        '7951': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5429:
      type: Mul
      input:
        ret2429.1: {}
        '7949': {}
      output:
        ret2445.1: {}
      attr:
        algorithm: mul
    aten::size_2308:
      type: Shape
      input:
        ret2429.1: {}
      output:
        '7955': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_234:
      type: Div
      input:
        '7955': {}
        '495': {}
      output:
        ret2447.1: {}
      attr:
        algorithm: div
    aten::slice_2309:
      type: Slice
      input:
        ret2429.1: {}
        ret2447.1: {}
      output:
        ret2448.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2310:
      type: Slice
      input:
        ret2429.1: {}
        ret2447.1: {}
      output:
        ret2449.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5436:
      type: Neg
      input:
        ret2449.1: {}
        aten::neg_5436_mul_val: {}
      output:
        ret2450.1: {}
      attr:
        algorithm: mul
    aten::cat_2511:
      type: Concat
      input:
        ret2450.1: {}
        ret2448.1: {}
      output:
        ret2451.1: {}
      attr:
        axis: -1
    aten::mul_5433:
      type: Mul
      input:
        ret2451.1: {}
        '7951': {}
      output:
        ret2452.1: {}
      attr:
        algorithm: mul
    aten::add_3272:
      type: Add
      input:
        ret2445.1: {}
        ret2452.1: {}
      output:
        args115.1: {}
    aten::mul_5431:
      type: Mul
      input:
        ret2435.1: {}
        '7949': {}
      output:
        ret2453.1: {}
      attr:
        algorithm: mul
    aten::size_2311:
      type: Shape
      input:
        ret2435.1: {}
      output:
        '7977': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_235:
      type: Div
      input:
        '7977': {}
        '495': {}
      output:
        ret2455.1: {}
      attr:
        algorithm: div
    aten::slice_2312:
      type: Slice
      input:
        ret2435.1: {}
        ret2455.1: {}
      output:
        ret2456.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2313:
      type: Slice
      input:
        ret2435.1: {}
        ret2455.1: {}
      output:
        ret2457.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5438:
      type: Neg
      input:
        ret2457.1: {}
        aten::neg_5438_mul_val: {}
      output:
        ret2458.1: {}
      attr:
        algorithm: mul
    aten::cat_2512:
      type: Concat
      input:
        ret2458.1: {}
        ret2456.1: {}
      output:
        ret2459.1: {}
      attr:
        axis: -1
    aten::mul_5434:
      type: Mul
      input:
        ret2459.1: {}
        '7951': {}
      output:
        ret2460.1: {}
      attr:
        algorithm: mul
    aten::add_3273:
      type: Add
      input:
        ret2453.1: {}
        ret2460.1: {}
      output:
        '7995': {}
    aten::cat_2924:
      type: Concat
      input:
        x57.1: {}
        '7995': {}
      output:
        ret2461.1: {}
      attr:
        axis: 2
    aten::cat_2925:
      type: Concat
      input:
        x58.1: {}
        ret2441.1: {}
      output:
        ret2462.1: {}
      attr:
        axis: 2
    aten::transpose_2314:
      type: Reorder
      input:
        ret2461.1: {}
      output:
        ret2463.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5441:
      type: Matmul
      input:
        args115.1: {}
        ret2463.1: {}
      output:
        ret2466.1: {}
    aten::div_285:
      type: Div
      input:
        ret2466.1: {}
        '496': {}
      output:
        ret2467.1: {}
      attr:
        algorithm: div
    aten::add_3274:
      type: Add
      input:
        ret2467.1: {}
        attention_mask0.1: {}
      output:
        attn_weights58.1: {}
    aten::max_325:
      type: Max
      input:
        attn_weights58.1: {}
        '497': {}
      output:
        input58.1: {}
    aten::softmax_2438:
      type: Softmax
      input:
        input58.1: {}
      output:
        '8017': {}
      attr:
        axis: -1
    aten::matmul_5444:
      type: Matmul
      input:
        '8017': {}
        ret2462.1: {}
      output:
        ret2470.1: {}
    aten::transpose_2926:
      type: Reorder
      input:
        ret2470.1: {}
      output:
        ret2471.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5446:
      type: Reshape
      input:
        ret2471.1: {}
        '7879': {}
        '7881': {}
      output:
        ret2472.1: {}
    aten::mul_800:
      type: Mul
      input:
        ret2472.1: {}
        '962': {}
      output:
        ret2473.1: {}
      attr:
        algorithm: mul
    aten::linear_3642:
      type: InnerProduct
      input:
        ret2473.1: {}
        '963': {}
        aten::linear_3642_bias: {}
      output:
        ret2476.1: {}
    aten::add_3275:
      type: Add
      input:
        ret2415.1: {}
        ret2476.1: {}
      output:
        ret2477.1: {}
    aten::pow_2927:
      type: Pow
      input:
        ret2477.1: {}
        aten::pow_2927_other: {}
      output:
        ret2478.1: {}
    aten::mean_1032:
      type: ReduceMean
      input:
        ret2478.1: {}
      output:
        ret2479.1: {}
    aten::add_67:
      type: Add
      input:
        ret2479.1: {}
        '485': {}
      output:
        ret2480.1: {}
    aten::rsqrt_5449:
      type: Rsqrt
      input:
        ret2480.1: {}
      output:
        ret2481.1: {}
    aten::mul_5448:
      type: Mul
      input:
        ret2477.1: {}
        ret2481.1: {}
      output:
        ret2482.1: {}
      attr:
        algorithm: mul
    aten::mul_802:
      type: Mul
      input:
        '964': {}
        ret2482.1: {}
      output:
        ret2483.1: {}
      attr:
        algorithm: mul
    aten::mul_803:
      type: Mul
      input:
        ret2483.1: {}
        '965': {}
      output:
        ret2484.1: {}
      attr:
        algorithm: mul
    aten::linear_3643:
      type: InnerProduct
      input:
        ret2484.1: {}
        '966': {}
        aten::linear_3643_bias: {}
      output:
        ret2487.1: {}
    aten::silu_5451:
      type: Swish
      input:
        ret2487.1: {}
      output:
        ret2488.1: {}
    aten::mul_805:
      type: Mul
      input:
        ret2483.1: {}
        '967': {}
      output:
        ret2489.1: {}
      attr:
        algorithm: mul
    aten::linear_3644:
      type: InnerProduct
      input:
        ret2489.1: {}
        '968': {}
        aten::linear_3644_bias: {}
      output:
        ret2492.1: {}
    aten::mul_5452:
      type: Mul
      input:
        ret2488.1: {}
        ret2492.1: {}
      output:
        ret2493.1: {}
      attr:
        algorithm: mul
    aten::mul_807:
      type: Mul
      input:
        ret2493.1: {}
        '969': {}
      output:
        ret2494.1: {}
      attr:
        algorithm: mul
    aten::linear_3645:
      type: InnerProduct
      input:
        ret2494.1: {}
        '970': {}
        aten::linear_3645_bias: {}
      output:
        ret2497.1: {}
    aten::add_3276:
      type: Add
      input:
        ret2477.1: {}
        ret2497.1: {}
      output:
        ret2498.1: {}
    aten::pow_2928:
      type: Pow
      input:
        ret2498.1: {}
        aten::pow_2928_other: {}
      output:
        ret2499.1: {}
    aten::mean_1033:
      type: ReduceMean
      input:
        ret2499.1: {}
      output:
        ret2500.1: {}
    aten::add_68:
      type: Add
      input:
        ret2500.1: {}
        '485': {}
      output:
        ret2501.1: {}
    aten::rsqrt_5456:
      type: Rsqrt
      input:
        ret2501.1: {}
      output:
        ret2502.1: {}
    aten::mul_5455:
      type: Mul
      input:
        ret2498.1: {}
        ret2502.1: {}
      output:
        ret2503.1: {}
      attr:
        algorithm: mul
    aten::mul_809:
      type: Mul
      input:
        '971': {}
        ret2503.1: {}
      output:
        ret2504.1: {}
      attr:
        algorithm: mul
    aten::size_3419:
      type: Shape
      input:
        ret2504.1: {}
      output:
        '8102': {}
      attr:
        start: 0
        end: 0
    aten::size_3277:
      type: Shape
      input:
        ret2504.1: {}
      output:
        '8104': {}
      attr:
        start: 1
        end: 1
    aten::mul_810:
      type: Mul
      input:
        ret2504.1: {}
        '972': {}
      output:
        ret2507.1: {}
      attr:
        algorithm: mul
    aten::linear_3646:
      type: InnerProduct
      input:
        ret2507.1: {}
        '973': {}
        aten::linear_3646_bias: {}
      output:
        ret2510.1: {}
    aten::view_5458:
      type: View
      input:
        ret2510.1: {}
        '8102': {}
        '8104': {}
      output:
        ret2511.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2929:
      type: Reorder
      input:
        ret2511.1: {}
      output:
        ret2512.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_812:
      type: Mul
      input:
        ret2504.1: {}
        '974': {}
      output:
        ret2513.1: {}
      attr:
        algorithm: mul
    aten::linear_3647:
      type: InnerProduct
      input:
        ret2513.1: {}
        '975': {}
        aten::linear_3647_bias: {}
      output:
        ret2516.1: {}
    aten::view_5459:
      type: View
      input:
        ret2516.1: {}
        '8102': {}
        '8104': {}
      output:
        ret2517.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2930:
      type: Reorder
      input:
        ret2517.1: {}
      output:
        ret2518.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_814:
      type: Mul
      input:
        ret2504.1: {}
        '976': {}
      output:
        ret2519.1: {}
      attr:
        algorithm: mul
    aten::linear_3648:
      type: InnerProduct
      input:
        ret2519.1: {}
        '977': {}
        aten::linear_3648_bias: {}
      output:
        ret2522.1: {}
    aten::view_5460:
      type: View
      input:
        ret2522.1: {}
        '8102': {}
        '8104': {}
      output:
        ret2523.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2931:
      type: Reorder
      input:
        ret2523.1: {}
      output:
        ret2524.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2932:
      type: Shape
      input:
        ret2518.1: {}
      output:
        '8150': {}
      attr:
        start: 2
        end: 2
    aten::size_2933:
      type: Shape
      input:
        x59.1: {}
      output:
        '8153': {}
      attr:
        start: 2
        end: 2
    aten::add_3377:
      type: Add
      input:
        '8150': {}
        '8153': {}
      output:
        seq_len29.1: {}
    aten::slice_126:
      type: Slice
      input:
        '493': {}
        seq_len29.1: {}
      output:
        '8160': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_166:
      type: Slice
      input:
        '494': {}
        seq_len29.1: {}
      output:
        '8162': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2934:
      type: Shape
      input:
        ret2512.1: {}
      output:
        '8164': {}
      attr:
        start: 2
        end: 2
    aten::add_3278:
      type: Add
      input:
        '8164': {}
        '8153': {}
      output:
        '8169': {}
    aten::slice_2935:
      type: Slice
      input:
        '8160': {}
        '8153': {}
        '8169': {}
      output:
        '8171': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2315:
      type: Slice
      input:
        '8171': {}
      output:
        '8172': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2936:
      type: Slice
      input:
        '8162': {}
        '8153': {}
        '8169': {}
      output:
        '8173': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2316:
      type: Slice
      input:
        '8173': {}
      output:
        '8174': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5461:
      type: Mul
      input:
        ret2512.1: {}
        '8172': {}
      output:
        ret2528.1: {}
      attr:
        algorithm: mul
    aten::size_2317:
      type: Shape
      input:
        ret2512.1: {}
      output:
        '8178': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_236:
      type: Div
      input:
        '8178': {}
        '495': {}
      output:
        ret2530.1: {}
      attr:
        algorithm: div
    aten::slice_2318:
      type: Slice
      input:
        ret2512.1: {}
        ret2530.1: {}
      output:
        ret2531.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2319:
      type: Slice
      input:
        ret2512.1: {}
        ret2530.1: {}
      output:
        ret2532.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5468:
      type: Neg
      input:
        ret2532.1: {}
        aten::neg_5468_mul_val: {}
      output:
        ret2533.1: {}
      attr:
        algorithm: mul
    aten::cat_2513:
      type: Concat
      input:
        ret2533.1: {}
        ret2531.1: {}
      output:
        ret2534.1: {}
      attr:
        axis: -1
    aten::mul_5465:
      type: Mul
      input:
        ret2534.1: {}
        '8174': {}
      output:
        ret2535.1: {}
      attr:
        algorithm: mul
    aten::add_3279:
      type: Add
      input:
        ret2528.1: {}
        ret2535.1: {}
      output:
        args119.1: {}
    aten::mul_5463:
      type: Mul
      input:
        ret2518.1: {}
        '8172': {}
      output:
        ret2536.1: {}
      attr:
        algorithm: mul
    aten::size_2320:
      type: Shape
      input:
        ret2518.1: {}
      output:
        '8200': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_237:
      type: Div
      input:
        '8200': {}
        '495': {}
      output:
        ret2538.1: {}
      attr:
        algorithm: div
    aten::slice_2321:
      type: Slice
      input:
        ret2518.1: {}
        ret2538.1: {}
      output:
        ret2539.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2322:
      type: Slice
      input:
        ret2518.1: {}
        ret2538.1: {}
      output:
        ret2540.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5470:
      type: Neg
      input:
        ret2540.1: {}
        aten::neg_5470_mul_val: {}
      output:
        ret2541.1: {}
      attr:
        algorithm: mul
    aten::cat_2514:
      type: Concat
      input:
        ret2541.1: {}
        ret2539.1: {}
      output:
        ret2542.1: {}
      attr:
        axis: -1
    aten::mul_5466:
      type: Mul
      input:
        ret2542.1: {}
        '8174': {}
      output:
        ret2543.1: {}
      attr:
        algorithm: mul
    aten::add_3280:
      type: Add
      input:
        ret2536.1: {}
        ret2543.1: {}
      output:
        '8218': {}
    aten::cat_2937:
      type: Concat
      input:
        x59.1: {}
        '8218': {}
      output:
        ret2544.1: {}
      attr:
        axis: 2
    aten::cat_2938:
      type: Concat
      input:
        x60.1: {}
        ret2524.1: {}
      output:
        ret2545.1: {}
      attr:
        axis: 2
    aten::transpose_2323:
      type: Reorder
      input:
        ret2544.1: {}
      output:
        ret2546.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5473:
      type: Matmul
      input:
        args119.1: {}
        ret2546.1: {}
      output:
        ret2549.1: {}
    aten::div_286:
      type: Div
      input:
        ret2549.1: {}
        '496': {}
      output:
        ret2550.1: {}
      attr:
        algorithm: div
    aten::add_3281:
      type: Add
      input:
        ret2550.1: {}
        attention_mask0.1: {}
      output:
        attn_weights60.1: {}
    aten::max_326:
      type: Max
      input:
        attn_weights60.1: {}
        '497': {}
      output:
        input60.1: {}
    aten::softmax_2439:
      type: Softmax
      input:
        input60.1: {}
      output:
        '8240': {}
      attr:
        axis: -1
    aten::matmul_5476:
      type: Matmul
      input:
        '8240': {}
        ret2545.1: {}
      output:
        ret2553.1: {}
    aten::transpose_2939:
      type: Reorder
      input:
        ret2553.1: {}
      output:
        ret2554.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5478:
      type: Reshape
      input:
        ret2554.1: {}
        '8102': {}
        '8104': {}
      output:
        ret2555.1: {}
    aten::mul_816:
      type: Mul
      input:
        ret2555.1: {}
        '978': {}
      output:
        ret2556.1: {}
      attr:
        algorithm: mul
    aten::linear_3649:
      type: InnerProduct
      input:
        ret2556.1: {}
        '979': {}
        aten::linear_3649_bias: {}
      output:
        ret2559.1: {}
    aten::add_3282:
      type: Add
      input:
        ret2498.1: {}
        ret2559.1: {}
      output:
        ret2560.1: {}
    aten::pow_2940:
      type: Pow
      input:
        ret2560.1: {}
        aten::pow_2940_other: {}
      output:
        ret2561.1: {}
    aten::mean_1034:
      type: ReduceMean
      input:
        ret2561.1: {}
      output:
        ret2562.1: {}
    aten::add_69:
      type: Add
      input:
        ret2562.1: {}
        '485': {}
      output:
        ret2563.1: {}
    aten::rsqrt_5481:
      type: Rsqrt
      input:
        ret2563.1: {}
      output:
        ret2564.1: {}
    aten::mul_5480:
      type: Mul
      input:
        ret2560.1: {}
        ret2564.1: {}
      output:
        ret2565.1: {}
      attr:
        algorithm: mul
    aten::mul_818:
      type: Mul
      input:
        '980': {}
        ret2565.1: {}
      output:
        ret2566.1: {}
      attr:
        algorithm: mul
    aten::mul_819:
      type: Mul
      input:
        ret2566.1: {}
        '981': {}
      output:
        ret2567.1: {}
      attr:
        algorithm: mul
    aten::linear_3650:
      type: InnerProduct
      input:
        ret2567.1: {}
        '982': {}
        aten::linear_3650_bias: {}
      output:
        ret2570.1: {}
    aten::silu_5483:
      type: Swish
      input:
        ret2570.1: {}
      output:
        ret2571.1: {}
    aten::mul_821:
      type: Mul
      input:
        ret2566.1: {}
        '983': {}
      output:
        ret2572.1: {}
      attr:
        algorithm: mul
    aten::linear_3651:
      type: InnerProduct
      input:
        ret2572.1: {}
        '984': {}
        aten::linear_3651_bias: {}
      output:
        ret2575.1: {}
    aten::mul_5484:
      type: Mul
      input:
        ret2571.1: {}
        ret2575.1: {}
      output:
        ret2576.1: {}
      attr:
        algorithm: mul
    aten::mul_823:
      type: Mul
      input:
        ret2576.1: {}
        '985': {}
      output:
        ret2577.1: {}
      attr:
        algorithm: mul
    aten::linear_3652:
      type: InnerProduct
      input:
        ret2577.1: {}
        '986': {}
        aten::linear_3652_bias: {}
      output:
        ret2580.1: {}
    aten::add_3283:
      type: Add
      input:
        ret2560.1: {}
        ret2580.1: {}
      output:
        ret2581.1: {}
    aten::pow_2941:
      type: Pow
      input:
        ret2581.1: {}
        aten::pow_2941_other: {}
      output:
        ret2582.1: {}
    aten::mean_1035:
      type: ReduceMean
      input:
        ret2582.1: {}
      output:
        ret2583.1: {}
    aten::add_70:
      type: Add
      input:
        ret2583.1: {}
        '485': {}
      output:
        ret2584.1: {}
    aten::rsqrt_5488:
      type: Rsqrt
      input:
        ret2584.1: {}
      output:
        ret2585.1: {}
    aten::mul_5487:
      type: Mul
      input:
        ret2581.1: {}
        ret2585.1: {}
      output:
        ret2586.1: {}
      attr:
        algorithm: mul
    aten::mul_825:
      type: Mul
      input:
        '987': {}
        ret2586.1: {}
      output:
        ret2587.1: {}
      attr:
        algorithm: mul
    aten::size_3420:
      type: Shape
      input:
        ret2587.1: {}
      output:
        '8325': {}
      attr:
        start: 0
        end: 0
    aten::size_3284:
      type: Shape
      input:
        ret2587.1: {}
      output:
        '8327': {}
      attr:
        start: 1
        end: 1
    aten::mul_826:
      type: Mul
      input:
        ret2587.1: {}
        '988': {}
      output:
        ret2590.1: {}
      attr:
        algorithm: mul
    aten::linear_3653:
      type: InnerProduct
      input:
        ret2590.1: {}
        '989': {}
        aten::linear_3653_bias: {}
      output:
        ret2593.1: {}
    aten::view_5490:
      type: View
      input:
        ret2593.1: {}
        '8325': {}
        '8327': {}
      output:
        ret2594.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2942:
      type: Reorder
      input:
        ret2594.1: {}
      output:
        ret2595.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_828:
      type: Mul
      input:
        ret2587.1: {}
        '990': {}
      output:
        ret2596.1: {}
      attr:
        algorithm: mul
    aten::linear_3654:
      type: InnerProduct
      input:
        ret2596.1: {}
        '991': {}
        aten::linear_3654_bias: {}
      output:
        ret2599.1: {}
    aten::view_5491:
      type: View
      input:
        ret2599.1: {}
        '8325': {}
        '8327': {}
      output:
        ret2600.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2943:
      type: Reorder
      input:
        ret2600.1: {}
      output:
        ret2601.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_830:
      type: Mul
      input:
        ret2587.1: {}
        '992': {}
      output:
        ret2602.1: {}
      attr:
        algorithm: mul
    aten::linear_3655:
      type: InnerProduct
      input:
        ret2602.1: {}
        '993': {}
        aten::linear_3655_bias: {}
      output:
        ret2605.1: {}
    aten::view_5492:
      type: View
      input:
        ret2605.1: {}
        '8325': {}
        '8327': {}
      output:
        ret2606.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2944:
      type: Reorder
      input:
        ret2606.1: {}
      output:
        ret2607.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2945:
      type: Shape
      input:
        ret2601.1: {}
      output:
        '8373': {}
      attr:
        start: 2
        end: 2
    aten::size_2946:
      type: Shape
      input:
        x61.1: {}
      output:
        '8376': {}
      attr:
        start: 2
        end: 2
    aten::add_3378:
      type: Add
      input:
        '8373': {}
        '8376': {}
      output:
        seq_len30.1: {}
    aten::slice_127:
      type: Slice
      input:
        '493': {}
        seq_len30.1: {}
      output:
        '8383': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_167:
      type: Slice
      input:
        '494': {}
        seq_len30.1: {}
      output:
        '8385': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2947:
      type: Shape
      input:
        ret2595.1: {}
      output:
        '8387': {}
      attr:
        start: 2
        end: 2
    aten::add_3285:
      type: Add
      input:
        '8387': {}
        '8376': {}
      output:
        '8392': {}
    aten::slice_2948:
      type: Slice
      input:
        '8383': {}
        '8376': {}
        '8392': {}
      output:
        '8394': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2324:
      type: Slice
      input:
        '8394': {}
      output:
        '8395': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2949:
      type: Slice
      input:
        '8385': {}
        '8376': {}
        '8392': {}
      output:
        '8396': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2325:
      type: Slice
      input:
        '8396': {}
      output:
        '8397': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5493:
      type: Mul
      input:
        ret2595.1: {}
        '8395': {}
      output:
        ret2611.1: {}
      attr:
        algorithm: mul
    aten::size_2326:
      type: Shape
      input:
        ret2595.1: {}
      output:
        '8401': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_238:
      type: Div
      input:
        '8401': {}
        '495': {}
      output:
        ret2613.1: {}
      attr:
        algorithm: div
    aten::slice_2327:
      type: Slice
      input:
        ret2595.1: {}
        ret2613.1: {}
      output:
        ret2614.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2328:
      type: Slice
      input:
        ret2595.1: {}
        ret2613.1: {}
      output:
        ret2615.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5500:
      type: Neg
      input:
        ret2615.1: {}
        aten::neg_5500_mul_val: {}
      output:
        ret2616.1: {}
      attr:
        algorithm: mul
    aten::cat_2515:
      type: Concat
      input:
        ret2616.1: {}
        ret2614.1: {}
      output:
        ret2617.1: {}
      attr:
        axis: -1
    aten::mul_5497:
      type: Mul
      input:
        ret2617.1: {}
        '8397': {}
      output:
        ret2618.1: {}
      attr:
        algorithm: mul
    aten::add_3286:
      type: Add
      input:
        ret2611.1: {}
        ret2618.1: {}
      output:
        args123.1: {}
    aten::mul_5495:
      type: Mul
      input:
        ret2601.1: {}
        '8395': {}
      output:
        ret2619.1: {}
      attr:
        algorithm: mul
    aten::size_2329:
      type: Shape
      input:
        ret2601.1: {}
      output:
        '8423': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_239:
      type: Div
      input:
        '8423': {}
        '495': {}
      output:
        ret2621.1: {}
      attr:
        algorithm: div
    aten::slice_2330:
      type: Slice
      input:
        ret2601.1: {}
        ret2621.1: {}
      output:
        ret2622.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2331:
      type: Slice
      input:
        ret2601.1: {}
        ret2621.1: {}
      output:
        ret2623.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5502:
      type: Neg
      input:
        ret2623.1: {}
        aten::neg_5502_mul_val: {}
      output:
        ret2624.1: {}
      attr:
        algorithm: mul
    aten::cat_2516:
      type: Concat
      input:
        ret2624.1: {}
        ret2622.1: {}
      output:
        ret2625.1: {}
      attr:
        axis: -1
    aten::mul_5498:
      type: Mul
      input:
        ret2625.1: {}
        '8397': {}
      output:
        ret2626.1: {}
      attr:
        algorithm: mul
    aten::add_3287:
      type: Add
      input:
        ret2619.1: {}
        ret2626.1: {}
      output:
        '8441': {}
    aten::cat_2950:
      type: Concat
      input:
        x61.1: {}
        '8441': {}
      output:
        ret2627.1: {}
      attr:
        axis: 2
    aten::cat_2951:
      type: Concat
      input:
        x62.1: {}
        ret2607.1: {}
      output:
        ret2628.1: {}
      attr:
        axis: 2
    aten::transpose_2332:
      type: Reorder
      input:
        ret2627.1: {}
      output:
        ret2629.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5505:
      type: Matmul
      input:
        args123.1: {}
        ret2629.1: {}
      output:
        ret2632.1: {}
    aten::div_287:
      type: Div
      input:
        ret2632.1: {}
        '496': {}
      output:
        ret2633.1: {}
      attr:
        algorithm: div
    aten::add_3288:
      type: Add
      input:
        ret2633.1: {}
        attention_mask0.1: {}
      output:
        attn_weights62.1: {}
    aten::max_327:
      type: Max
      input:
        attn_weights62.1: {}
        '497': {}
      output:
        input62.1: {}
    aten::softmax_2440:
      type: Softmax
      input:
        input62.1: {}
      output:
        '8463': {}
      attr:
        axis: -1
    aten::matmul_5508:
      type: Matmul
      input:
        '8463': {}
        ret2628.1: {}
      output:
        ret2636.1: {}
    aten::transpose_2952:
      type: Reorder
      input:
        ret2636.1: {}
      output:
        ret2637.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5510:
      type: Reshape
      input:
        ret2637.1: {}
        '8325': {}
        '8327': {}
      output:
        ret2638.1: {}
    aten::mul_832:
      type: Mul
      input:
        ret2638.1: {}
        '994': {}
      output:
        ret2639.1: {}
      attr:
        algorithm: mul
    aten::linear_3656:
      type: InnerProduct
      input:
        ret2639.1: {}
        '995': {}
        aten::linear_3656_bias: {}
      output:
        ret2642.1: {}
    aten::add_3289:
      type: Add
      input:
        ret2581.1: {}
        ret2642.1: {}
      output:
        ret2643.1: {}
    aten::pow_2953:
      type: Pow
      input:
        ret2643.1: {}
        aten::pow_2953_other: {}
      output:
        ret2644.1: {}
    aten::mean_1036:
      type: ReduceMean
      input:
        ret2644.1: {}
      output:
        ret2645.1: {}
    aten::add_71:
      type: Add
      input:
        ret2645.1: {}
        '485': {}
      output:
        ret2646.1: {}
    aten::rsqrt_5513:
      type: Rsqrt
      input:
        ret2646.1: {}
      output:
        ret2647.1: {}
    aten::mul_5512:
      type: Mul
      input:
        ret2643.1: {}
        ret2647.1: {}
      output:
        ret2648.1: {}
      attr:
        algorithm: mul
    aten::mul_834:
      type: Mul
      input:
        '996': {}
        ret2648.1: {}
      output:
        ret2649.1: {}
      attr:
        algorithm: mul
    aten::mul_835:
      type: Mul
      input:
        ret2649.1: {}
        '997': {}
      output:
        ret2650.1: {}
      attr:
        algorithm: mul
    aten::linear_3657:
      type: InnerProduct
      input:
        ret2650.1: {}
        '998': {}
        aten::linear_3657_bias: {}
      output:
        ret2653.1: {}
    aten::silu_5515:
      type: Swish
      input:
        ret2653.1: {}
      output:
        ret2654.1: {}
    aten::mul_837:
      type: Mul
      input:
        ret2649.1: {}
        '999': {}
      output:
        ret2655.1: {}
      attr:
        algorithm: mul
    aten::linear_3658:
      type: InnerProduct
      input:
        ret2655.1: {}
        '1000': {}
        aten::linear_3658_bias: {}
      output:
        ret2658.1: {}
    aten::mul_5516:
      type: Mul
      input:
        ret2654.1: {}
        ret2658.1: {}
      output:
        ret2659.1: {}
      attr:
        algorithm: mul
    aten::mul_839:
      type: Mul
      input:
        ret2659.1: {}
        '1001': {}
      output:
        ret2660.1: {}
      attr:
        algorithm: mul
    aten::linear_3659:
      type: InnerProduct
      input:
        ret2660.1: {}
        '1002': {}
        aten::linear_3659_bias: {}
      output:
        ret2663.1: {}
    aten::add_3290:
      type: Add
      input:
        ret2643.1: {}
        ret2663.1: {}
      output:
        ret2664.1: {}
    aten::pow_2954:
      type: Pow
      input:
        ret2664.1: {}
        aten::pow_2954_other: {}
      output:
        ret2665.1: {}
    aten::mean_1037:
      type: ReduceMean
      input:
        ret2665.1: {}
      output:
        ret2666.1: {}
    aten::add_72:
      type: Add
      input:
        ret2666.1: {}
        '485': {}
      output:
        ret2667.1: {}
    aten::rsqrt_5520:
      type: Rsqrt
      input:
        ret2667.1: {}
      output:
        ret2668.1: {}
    aten::mul_5519:
      type: Mul
      input:
        ret2664.1: {}
        ret2668.1: {}
      output:
        ret2669.1: {}
      attr:
        algorithm: mul
    aten::mul_841:
      type: Mul
      input:
        '1003': {}
        ret2669.1: {}
      output:
        ret2670.1: {}
      attr:
        algorithm: mul
    aten::size_3421:
      type: Shape
      input:
        ret2670.1: {}
      output:
        '8548': {}
      attr:
        start: 0
        end: 0
    aten::size_3291:
      type: Shape
      input:
        ret2670.1: {}
      output:
        '8550': {}
      attr:
        start: 1
        end: 1
    aten::mul_842:
      type: Mul
      input:
        ret2670.1: {}
        '1004': {}
      output:
        ret2673.1: {}
      attr:
        algorithm: mul
    aten::linear_3660:
      type: InnerProduct
      input:
        ret2673.1: {}
        '1005': {}
        aten::linear_3660_bias: {}
      output:
        ret2676.1: {}
    aten::view_5522:
      type: View
      input:
        ret2676.1: {}
        '8548': {}
        '8550': {}
      output:
        ret2677.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2955:
      type: Reorder
      input:
        ret2677.1: {}
      output:
        ret2678.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_844:
      type: Mul
      input:
        ret2670.1: {}
        '1006': {}
      output:
        ret2679.1: {}
      attr:
        algorithm: mul
    aten::linear_3661:
      type: InnerProduct
      input:
        ret2679.1: {}
        '1007': {}
        aten::linear_3661_bias: {}
      output:
        ret2682.1: {}
    aten::view_5523:
      type: View
      input:
        ret2682.1: {}
        '8548': {}
        '8550': {}
      output:
        ret2683.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2956:
      type: Reorder
      input:
        ret2683.1: {}
      output:
        ret2684.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_846:
      type: Mul
      input:
        ret2670.1: {}
        '1008': {}
      output:
        ret2685.1: {}
      attr:
        algorithm: mul
    aten::linear_3662:
      type: InnerProduct
      input:
        ret2685.1: {}
        '1009': {}
        aten::linear_3662_bias: {}
      output:
        ret2688.1: {}
    aten::view_5524:
      type: View
      input:
        ret2688.1: {}
        '8548': {}
        '8550': {}
      output:
        ret2689.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2957:
      type: Reorder
      input:
        ret2689.1: {}
      output:
        ret2690.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2958:
      type: Shape
      input:
        ret2684.1: {}
      output:
        '8596': {}
      attr:
        start: 2
        end: 2
    aten::size_2959:
      type: Shape
      input:
        x63.1: {}
      output:
        '8599': {}
      attr:
        start: 2
        end: 2
    aten::add_3379:
      type: Add
      input:
        '8596': {}
        '8599': {}
      output:
        seq_len31.1: {}
    aten::slice_128:
      type: Slice
      input:
        '493': {}
        seq_len31.1: {}
      output:
        '8606': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_168:
      type: Slice
      input:
        '494': {}
        seq_len31.1: {}
      output:
        '8608': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2960:
      type: Shape
      input:
        ret2678.1: {}
      output:
        '8610': {}
      attr:
        start: 2
        end: 2
    aten::add_3292:
      type: Add
      input:
        '8610': {}
        '8599': {}
      output:
        '8615': {}
    aten::slice_2961:
      type: Slice
      input:
        '8606': {}
        '8599': {}
        '8615': {}
      output:
        '8617': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2333:
      type: Slice
      input:
        '8617': {}
      output:
        '8618': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2962:
      type: Slice
      input:
        '8608': {}
        '8599': {}
        '8615': {}
      output:
        '8619': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2334:
      type: Slice
      input:
        '8619': {}
      output:
        '8620': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5525:
      type: Mul
      input:
        ret2678.1: {}
        '8618': {}
      output:
        ret2694.1: {}
      attr:
        algorithm: mul
    aten::size_2335:
      type: Shape
      input:
        ret2678.1: {}
      output:
        '8624': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_240:
      type: Div
      input:
        '8624': {}
        '495': {}
      output:
        ret2696.1: {}
      attr:
        algorithm: div
    aten::slice_2336:
      type: Slice
      input:
        ret2678.1: {}
        ret2696.1: {}
      output:
        ret2697.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2337:
      type: Slice
      input:
        ret2678.1: {}
        ret2696.1: {}
      output:
        ret2698.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5532:
      type: Neg
      input:
        ret2698.1: {}
        aten::neg_5532_mul_val: {}
      output:
        ret2699.1: {}
      attr:
        algorithm: mul
    aten::cat_2517:
      type: Concat
      input:
        ret2699.1: {}
        ret2697.1: {}
      output:
        ret2700.1: {}
      attr:
        axis: -1
    aten::mul_5529:
      type: Mul
      input:
        ret2700.1: {}
        '8620': {}
      output:
        ret2701.1: {}
      attr:
        algorithm: mul
    aten::add_3293:
      type: Add
      input:
        ret2694.1: {}
        ret2701.1: {}
      output:
        args127.1: {}
    aten::mul_5527:
      type: Mul
      input:
        ret2684.1: {}
        '8618': {}
      output:
        ret2702.1: {}
      attr:
        algorithm: mul
    aten::size_2338:
      type: Shape
      input:
        ret2684.1: {}
      output:
        '8646': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_241:
      type: Div
      input:
        '8646': {}
        '495': {}
      output:
        ret2704.1: {}
      attr:
        algorithm: div
    aten::slice_2339:
      type: Slice
      input:
        ret2684.1: {}
        ret2704.1: {}
      output:
        ret2705.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2340:
      type: Slice
      input:
        ret2684.1: {}
        ret2704.1: {}
      output:
        ret2706.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5534:
      type: Neg
      input:
        ret2706.1: {}
        aten::neg_5534_mul_val: {}
      output:
        ret2707.1: {}
      attr:
        algorithm: mul
    aten::cat_2518:
      type: Concat
      input:
        ret2707.1: {}
        ret2705.1: {}
      output:
        ret2708.1: {}
      attr:
        axis: -1
    aten::mul_5530:
      type: Mul
      input:
        ret2708.1: {}
        '8620': {}
      output:
        ret2709.1: {}
      attr:
        algorithm: mul
    aten::add_3294:
      type: Add
      input:
        ret2702.1: {}
        ret2709.1: {}
      output:
        '8664': {}
    aten::cat_2963:
      type: Concat
      input:
        x63.1: {}
        '8664': {}
      output:
        ret2710.1: {}
      attr:
        axis: 2
    aten::cat_2964:
      type: Concat
      input:
        x64.1: {}
        ret2690.1: {}
      output:
        ret2711.1: {}
      attr:
        axis: 2
    aten::transpose_2341:
      type: Reorder
      input:
        ret2710.1: {}
      output:
        ret2712.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5537:
      type: Matmul
      input:
        args127.1: {}
        ret2712.1: {}
      output:
        ret2715.1: {}
    aten::div_288:
      type: Div
      input:
        ret2715.1: {}
        '496': {}
      output:
        ret2716.1: {}
      attr:
        algorithm: div
    aten::add_3295:
      type: Add
      input:
        ret2716.1: {}
        attention_mask0.1: {}
      output:
        attn_weights64.1: {}
    aten::max_328:
      type: Max
      input:
        attn_weights64.1: {}
        '497': {}
      output:
        input64.1: {}
    aten::softmax_2441:
      type: Softmax
      input:
        input64.1: {}
      output:
        '8686': {}
      attr:
        axis: -1
    aten::matmul_5540:
      type: Matmul
      input:
        '8686': {}
        ret2711.1: {}
      output:
        ret2719.1: {}
    aten::transpose_2965:
      type: Reorder
      input:
        ret2719.1: {}
      output:
        ret2720.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5542:
      type: Reshape
      input:
        ret2720.1: {}
        '8548': {}
        '8550': {}
      output:
        ret2721.1: {}
    aten::mul_848:
      type: Mul
      input:
        ret2721.1: {}
        '1010': {}
      output:
        ret2722.1: {}
      attr:
        algorithm: mul
    aten::linear_3663:
      type: InnerProduct
      input:
        ret2722.1: {}
        '1011': {}
        aten::linear_3663_bias: {}
      output:
        ret2725.1: {}
    aten::add_3296:
      type: Add
      input:
        ret2664.1: {}
        ret2725.1: {}
      output:
        ret2726.1: {}
    aten::pow_2966:
      type: Pow
      input:
        ret2726.1: {}
        aten::pow_2966_other: {}
      output:
        ret2727.1: {}
    aten::mean_1038:
      type: ReduceMean
      input:
        ret2727.1: {}
      output:
        ret2728.1: {}
    aten::add_73:
      type: Add
      input:
        ret2728.1: {}
        '485': {}
      output:
        ret2729.1: {}
    aten::rsqrt_5545:
      type: Rsqrt
      input:
        ret2729.1: {}
      output:
        ret2730.1: {}
    aten::mul_5544:
      type: Mul
      input:
        ret2726.1: {}
        ret2730.1: {}
      output:
        ret2731.1: {}
      attr:
        algorithm: mul
    aten::mul_850:
      type: Mul
      input:
        '1012': {}
        ret2731.1: {}
      output:
        ret2732.1: {}
      attr:
        algorithm: mul
    aten::mul_851:
      type: Mul
      input:
        ret2732.1: {}
        '1013': {}
      output:
        ret2733.1: {}
      attr:
        algorithm: mul
    aten::linear_3664:
      type: InnerProduct
      input:
        ret2733.1: {}
        '1014': {}
        aten::linear_3664_bias: {}
      output:
        ret2736.1: {}
    aten::silu_5547:
      type: Swish
      input:
        ret2736.1: {}
      output:
        ret2737.1: {}
    aten::mul_853:
      type: Mul
      input:
        ret2732.1: {}
        '1015': {}
      output:
        ret2738.1: {}
      attr:
        algorithm: mul
    aten::linear_3665:
      type: InnerProduct
      input:
        ret2738.1: {}
        '1016': {}
        aten::linear_3665_bias: {}
      output:
        ret2741.1: {}
    aten::mul_5548:
      type: Mul
      input:
        ret2737.1: {}
        ret2741.1: {}
      output:
        ret2742.1: {}
      attr:
        algorithm: mul
    aten::mul_855:
      type: Mul
      input:
        ret2742.1: {}
        '1017': {}
      output:
        ret2743.1: {}
      attr:
        algorithm: mul
    aten::linear_3666:
      type: InnerProduct
      input:
        ret2743.1: {}
        '1018': {}
        aten::linear_3666_bias: {}
      output:
        ret2746.1: {}
    aten::add_3297:
      type: Add
      input:
        ret2726.1: {}
        ret2746.1: {}
      output:
        ret2747.1: {}
    aten::pow_2967:
      type: Pow
      input:
        ret2747.1: {}
        aten::pow_2967_other: {}
      output:
        ret2748.1: {}
    aten::mean_1039:
      type: ReduceMean
      input:
        ret2748.1: {}
      output:
        ret2749.1: {}
    aten::add_74:
      type: Add
      input:
        ret2749.1: {}
        '485': {}
      output:
        ret2750.1: {}
    aten::rsqrt_5552:
      type: Rsqrt
      input:
        ret2750.1: {}
      output:
        ret2751.1: {}
    aten::mul_5551:
      type: Mul
      input:
        ret2747.1: {}
        ret2751.1: {}
      output:
        ret2752.1: {}
      attr:
        algorithm: mul
    aten::mul_857:
      type: Mul
      input:
        '1019': {}
        ret2752.1: {}
      output:
        ret2753.1: {}
      attr:
        algorithm: mul
    aten::size_3422:
      type: Shape
      input:
        ret2753.1: {}
      output:
        '8771': {}
      attr:
        start: 0
        end: 0
    aten::size_3298:
      type: Shape
      input:
        ret2753.1: {}
      output:
        '8773': {}
      attr:
        start: 1
        end: 1
    aten::mul_858:
      type: Mul
      input:
        ret2753.1: {}
        '1020': {}
      output:
        ret2756.1: {}
      attr:
        algorithm: mul
    aten::linear_3667:
      type: InnerProduct
      input:
        ret2756.1: {}
        '1021': {}
        aten::linear_3667_bias: {}
      output:
        ret2759.1: {}
    aten::view_5554:
      type: View
      input:
        ret2759.1: {}
        '8771': {}
        '8773': {}
      output:
        ret2760.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2968:
      type: Reorder
      input:
        ret2760.1: {}
      output:
        ret2761.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_860:
      type: Mul
      input:
        ret2753.1: {}
        '1022': {}
      output:
        ret2762.1: {}
      attr:
        algorithm: mul
    aten::linear_3668:
      type: InnerProduct
      input:
        ret2762.1: {}
        '1023': {}
        aten::linear_3668_bias: {}
      output:
        ret2765.1: {}
    aten::view_5555:
      type: View
      input:
        ret2765.1: {}
        '8771': {}
        '8773': {}
      output:
        ret2766.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2969:
      type: Reorder
      input:
        ret2766.1: {}
      output:
        ret2767.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_862:
      type: Mul
      input:
        ret2753.1: {}
        '1024': {}
      output:
        ret2768.1: {}
      attr:
        algorithm: mul
    aten::linear_3669:
      type: InnerProduct
      input:
        ret2768.1: {}
        '1025': {}
        aten::linear_3669_bias: {}
      output:
        ret2771.1: {}
    aten::view_5556:
      type: View
      input:
        ret2771.1: {}
        '8771': {}
        '8773': {}
      output:
        ret2772.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2970:
      type: Reorder
      input:
        ret2772.1: {}
      output:
        ret2773.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2971:
      type: Shape
      input:
        ret2767.1: {}
      output:
        '8819': {}
      attr:
        start: 2
        end: 2
    aten::size_2972:
      type: Shape
      input:
        x65.1: {}
      output:
        '8822': {}
      attr:
        start: 2
        end: 2
    aten::add_3380:
      type: Add
      input:
        '8819': {}
        '8822': {}
      output:
        seq_len32.1: {}
    aten::slice_129:
      type: Slice
      input:
        '493': {}
        seq_len32.1: {}
      output:
        '8829': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_169:
      type: Slice
      input:
        '494': {}
        seq_len32.1: {}
      output:
        '8831': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2973:
      type: Shape
      input:
        ret2761.1: {}
      output:
        '8833': {}
      attr:
        start: 2
        end: 2
    aten::add_3299:
      type: Add
      input:
        '8833': {}
        '8822': {}
      output:
        '8838': {}
    aten::slice_2974:
      type: Slice
      input:
        '8829': {}
        '8822': {}
        '8838': {}
      output:
        '8840': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2342:
      type: Slice
      input:
        '8840': {}
      output:
        '8841': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2975:
      type: Slice
      input:
        '8831': {}
        '8822': {}
        '8838': {}
      output:
        '8842': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2343:
      type: Slice
      input:
        '8842': {}
      output:
        '8843': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5557:
      type: Mul
      input:
        ret2761.1: {}
        '8841': {}
      output:
        ret2777.1: {}
      attr:
        algorithm: mul
    aten::size_2344:
      type: Shape
      input:
        ret2761.1: {}
      output:
        '8847': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_242:
      type: Div
      input:
        '8847': {}
        '495': {}
      output:
        ret2779.1: {}
      attr:
        algorithm: div
    aten::slice_2345:
      type: Slice
      input:
        ret2761.1: {}
        ret2779.1: {}
      output:
        ret2780.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2346:
      type: Slice
      input:
        ret2761.1: {}
        ret2779.1: {}
      output:
        ret2781.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5564:
      type: Neg
      input:
        ret2781.1: {}
        aten::neg_5564_mul_val: {}
      output:
        ret2782.1: {}
      attr:
        algorithm: mul
    aten::cat_2519:
      type: Concat
      input:
        ret2782.1: {}
        ret2780.1: {}
      output:
        ret2783.1: {}
      attr:
        axis: -1
    aten::mul_5561:
      type: Mul
      input:
        ret2783.1: {}
        '8843': {}
      output:
        ret2784.1: {}
      attr:
        algorithm: mul
    aten::add_3300:
      type: Add
      input:
        ret2777.1: {}
        ret2784.1: {}
      output:
        args131.1: {}
    aten::mul_5559:
      type: Mul
      input:
        ret2767.1: {}
        '8841': {}
      output:
        ret2785.1: {}
      attr:
        algorithm: mul
    aten::size_2347:
      type: Shape
      input:
        ret2767.1: {}
      output:
        '8869': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_243:
      type: Div
      input:
        '8869': {}
        '495': {}
      output:
        ret2787.1: {}
      attr:
        algorithm: div
    aten::slice_2348:
      type: Slice
      input:
        ret2767.1: {}
        ret2787.1: {}
      output:
        ret2788.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2349:
      type: Slice
      input:
        ret2767.1: {}
        ret2787.1: {}
      output:
        ret2789.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5566:
      type: Neg
      input:
        ret2789.1: {}
        aten::neg_5566_mul_val: {}
      output:
        ret2790.1: {}
      attr:
        algorithm: mul
    aten::cat_2520:
      type: Concat
      input:
        ret2790.1: {}
        ret2788.1: {}
      output:
        ret2791.1: {}
      attr:
        axis: -1
    aten::mul_5562:
      type: Mul
      input:
        ret2791.1: {}
        '8843': {}
      output:
        ret2792.1: {}
      attr:
        algorithm: mul
    aten::add_3301:
      type: Add
      input:
        ret2785.1: {}
        ret2792.1: {}
      output:
        '8887': {}
    aten::cat_2976:
      type: Concat
      input:
        x65.1: {}
        '8887': {}
      output:
        ret2793.1: {}
      attr:
        axis: 2
    aten::cat_2977:
      type: Concat
      input:
        x66.1: {}
        ret2773.1: {}
      output:
        ret2794.1: {}
      attr:
        axis: 2
    aten::transpose_2350:
      type: Reorder
      input:
        ret2793.1: {}
      output:
        ret2795.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5569:
      type: Matmul
      input:
        args131.1: {}
        ret2795.1: {}
      output:
        ret2798.1: {}
    aten::div_289:
      type: Div
      input:
        ret2798.1: {}
        '496': {}
      output:
        ret2799.1: {}
      attr:
        algorithm: div
    aten::add_3302:
      type: Add
      input:
        ret2799.1: {}
        attention_mask0.1: {}
      output:
        attn_weights66.1: {}
    aten::max_329:
      type: Max
      input:
        attn_weights66.1: {}
        '497': {}
      output:
        input66.1: {}
    aten::softmax_2442:
      type: Softmax
      input:
        input66.1: {}
      output:
        '8909': {}
      attr:
        axis: -1
    aten::matmul_5572:
      type: Matmul
      input:
        '8909': {}
        ret2794.1: {}
      output:
        ret2802.1: {}
    aten::transpose_2978:
      type: Reorder
      input:
        ret2802.1: {}
      output:
        ret2803.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5574:
      type: Reshape
      input:
        ret2803.1: {}
        '8771': {}
        '8773': {}
      output:
        ret2804.1: {}
    aten::mul_864:
      type: Mul
      input:
        ret2804.1: {}
        '1026': {}
      output:
        ret2805.1: {}
      attr:
        algorithm: mul
    aten::linear_3670:
      type: InnerProduct
      input:
        ret2805.1: {}
        '1027': {}
        aten::linear_3670_bias: {}
      output:
        ret2808.1: {}
    aten::add_3303:
      type: Add
      input:
        ret2747.1: {}
        ret2808.1: {}
      output:
        ret2809.1: {}
    aten::pow_2979:
      type: Pow
      input:
        ret2809.1: {}
        aten::pow_2979_other: {}
      output:
        ret2810.1: {}
    aten::mean_1040:
      type: ReduceMean
      input:
        ret2810.1: {}
      output:
        ret2811.1: {}
    aten::add_75:
      type: Add
      input:
        ret2811.1: {}
        '485': {}
      output:
        ret2812.1: {}
    aten::rsqrt_5577:
      type: Rsqrt
      input:
        ret2812.1: {}
      output:
        ret2813.1: {}
    aten::mul_5576:
      type: Mul
      input:
        ret2809.1: {}
        ret2813.1: {}
      output:
        ret2814.1: {}
      attr:
        algorithm: mul
    aten::mul_866:
      type: Mul
      input:
        '1028': {}
        ret2814.1: {}
      output:
        ret2815.1: {}
      attr:
        algorithm: mul
    aten::mul_867:
      type: Mul
      input:
        ret2815.1: {}
        '1029': {}
      output:
        ret2816.1: {}
      attr:
        algorithm: mul
    aten::linear_3671:
      type: InnerProduct
      input:
        ret2816.1: {}
        '1030': {}
        aten::linear_3671_bias: {}
      output:
        ret2819.1: {}
    aten::silu_5579:
      type: Swish
      input:
        ret2819.1: {}
      output:
        ret2820.1: {}
    aten::mul_869:
      type: Mul
      input:
        ret2815.1: {}
        '1031': {}
      output:
        ret2821.1: {}
      attr:
        algorithm: mul
    aten::linear_3672:
      type: InnerProduct
      input:
        ret2821.1: {}
        '1032': {}
        aten::linear_3672_bias: {}
      output:
        ret2824.1: {}
    aten::mul_5580:
      type: Mul
      input:
        ret2820.1: {}
        ret2824.1: {}
      output:
        ret2825.1: {}
      attr:
        algorithm: mul
    aten::mul_871:
      type: Mul
      input:
        ret2825.1: {}
        '1033': {}
      output:
        ret2826.1: {}
      attr:
        algorithm: mul
    aten::linear_3673:
      type: InnerProduct
      input:
        ret2826.1: {}
        '1034': {}
        aten::linear_3673_bias: {}
      output:
        ret2829.1: {}
    aten::add_3304:
      type: Add
      input:
        ret2809.1: {}
        ret2829.1: {}
      output:
        ret2830.1: {}
    aten::pow_2980:
      type: Pow
      input:
        ret2830.1: {}
        aten::pow_2980_other: {}
      output:
        ret2831.1: {}
    aten::mean_1041:
      type: ReduceMean
      input:
        ret2831.1: {}
      output:
        ret2832.1: {}
    aten::add_76:
      type: Add
      input:
        ret2832.1: {}
        '485': {}
      output:
        ret2833.1: {}
    aten::rsqrt_5584:
      type: Rsqrt
      input:
        ret2833.1: {}
      output:
        ret2834.1: {}
    aten::mul_5583:
      type: Mul
      input:
        ret2830.1: {}
        ret2834.1: {}
      output:
        ret2835.1: {}
      attr:
        algorithm: mul
    aten::mul_873:
      type: Mul
      input:
        '1035': {}
        ret2835.1: {}
      output:
        ret2836.1: {}
      attr:
        algorithm: mul
    aten::size_3423:
      type: Shape
      input:
        ret2836.1: {}
      output:
        '8994': {}
      attr:
        start: 0
        end: 0
    aten::size_3305:
      type: Shape
      input:
        ret2836.1: {}
      output:
        '8996': {}
      attr:
        start: 1
        end: 1
    aten::mul_874:
      type: Mul
      input:
        ret2836.1: {}
        '1036': {}
      output:
        ret2839.1: {}
      attr:
        algorithm: mul
    aten::linear_3674:
      type: InnerProduct
      input:
        ret2839.1: {}
        '1037': {}
        aten::linear_3674_bias: {}
      output:
        ret2842.1: {}
    aten::view_5586:
      type: View
      input:
        ret2842.1: {}
        '8994': {}
        '8996': {}
      output:
        ret2843.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2981:
      type: Reorder
      input:
        ret2843.1: {}
      output:
        ret2844.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_876:
      type: Mul
      input:
        ret2836.1: {}
        '1038': {}
      output:
        ret2845.1: {}
      attr:
        algorithm: mul
    aten::linear_3675:
      type: InnerProduct
      input:
        ret2845.1: {}
        '1039': {}
        aten::linear_3675_bias: {}
      output:
        ret2848.1: {}
    aten::view_5587:
      type: View
      input:
        ret2848.1: {}
        '8994': {}
        '8996': {}
      output:
        ret2849.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2982:
      type: Reorder
      input:
        ret2849.1: {}
      output:
        ret2850.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_878:
      type: Mul
      input:
        ret2836.1: {}
        '1040': {}
      output:
        ret2851.1: {}
      attr:
        algorithm: mul
    aten::linear_3676:
      type: InnerProduct
      input:
        ret2851.1: {}
        '1041': {}
        aten::linear_3676_bias: {}
      output:
        ret2854.1: {}
    aten::view_5588:
      type: View
      input:
        ret2854.1: {}
        '8994': {}
        '8996': {}
      output:
        ret2855.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2983:
      type: Reorder
      input:
        ret2855.1: {}
      output:
        ret2856.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2984:
      type: Shape
      input:
        ret2850.1: {}
      output:
        '9042': {}
      attr:
        start: 2
        end: 2
    aten::size_2985:
      type: Shape
      input:
        x67.1: {}
      output:
        '9045': {}
      attr:
        start: 2
        end: 2
    aten::add_3381:
      type: Add
      input:
        '9042': {}
        '9045': {}
      output:
        seq_len33.1: {}
    aten::slice_130:
      type: Slice
      input:
        '493': {}
        seq_len33.1: {}
      output:
        '9052': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_170:
      type: Slice
      input:
        '494': {}
        seq_len33.1: {}
      output:
        '9054': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2986:
      type: Shape
      input:
        ret2844.1: {}
      output:
        '9056': {}
      attr:
        start: 2
        end: 2
    aten::add_3306:
      type: Add
      input:
        '9056': {}
        '9045': {}
      output:
        '9061': {}
    aten::slice_2987:
      type: Slice
      input:
        '9052': {}
        '9045': {}
        '9061': {}
      output:
        '9063': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2351:
      type: Slice
      input:
        '9063': {}
      output:
        '9064': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_2988:
      type: Slice
      input:
        '9054': {}
        '9045': {}
        '9061': {}
      output:
        '9065': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2352:
      type: Slice
      input:
        '9065': {}
      output:
        '9066': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5589:
      type: Mul
      input:
        ret2844.1: {}
        '9064': {}
      output:
        ret2860.1: {}
      attr:
        algorithm: mul
    aten::size_2353:
      type: Shape
      input:
        ret2844.1: {}
      output:
        '9070': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_244:
      type: Div
      input:
        '9070': {}
        '495': {}
      output:
        ret2862.1: {}
      attr:
        algorithm: div
    aten::slice_2354:
      type: Slice
      input:
        ret2844.1: {}
        ret2862.1: {}
      output:
        ret2863.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2355:
      type: Slice
      input:
        ret2844.1: {}
        ret2862.1: {}
      output:
        ret2864.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5596:
      type: Neg
      input:
        ret2864.1: {}
        aten::neg_5596_mul_val: {}
      output:
        ret2865.1: {}
      attr:
        algorithm: mul
    aten::cat_2521:
      type: Concat
      input:
        ret2865.1: {}
        ret2863.1: {}
      output:
        ret2866.1: {}
      attr:
        axis: -1
    aten::mul_5593:
      type: Mul
      input:
        ret2866.1: {}
        '9066': {}
      output:
        ret2867.1: {}
      attr:
        algorithm: mul
    aten::add_3307:
      type: Add
      input:
        ret2860.1: {}
        ret2867.1: {}
      output:
        args135.1: {}
    aten::mul_5591:
      type: Mul
      input:
        ret2850.1: {}
        '9064': {}
      output:
        ret2868.1: {}
      attr:
        algorithm: mul
    aten::size_2356:
      type: Shape
      input:
        ret2850.1: {}
      output:
        '9092': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_245:
      type: Div
      input:
        '9092': {}
        '495': {}
      output:
        ret2870.1: {}
      attr:
        algorithm: div
    aten::slice_2357:
      type: Slice
      input:
        ret2850.1: {}
        ret2870.1: {}
      output:
        ret2871.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2358:
      type: Slice
      input:
        ret2850.1: {}
        ret2870.1: {}
      output:
        ret2872.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5598:
      type: Neg
      input:
        ret2872.1: {}
        aten::neg_5598_mul_val: {}
      output:
        ret2873.1: {}
      attr:
        algorithm: mul
    aten::cat_2522:
      type: Concat
      input:
        ret2873.1: {}
        ret2871.1: {}
      output:
        ret2874.1: {}
      attr:
        axis: -1
    aten::mul_5594:
      type: Mul
      input:
        ret2874.1: {}
        '9066': {}
      output:
        ret2875.1: {}
      attr:
        algorithm: mul
    aten::add_3308:
      type: Add
      input:
        ret2868.1: {}
        ret2875.1: {}
      output:
        '9110': {}
    aten::cat_2989:
      type: Concat
      input:
        x67.1: {}
        '9110': {}
      output:
        ret2876.1: {}
      attr:
        axis: 2
    aten::cat_2990:
      type: Concat
      input:
        x68.1: {}
        ret2856.1: {}
      output:
        ret2877.1: {}
      attr:
        axis: 2
    aten::transpose_2359:
      type: Reorder
      input:
        ret2876.1: {}
      output:
        ret2878.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5601:
      type: Matmul
      input:
        args135.1: {}
        ret2878.1: {}
      output:
        ret2881.1: {}
    aten::div_290:
      type: Div
      input:
        ret2881.1: {}
        '496': {}
      output:
        ret2882.1: {}
      attr:
        algorithm: div
    aten::add_3309:
      type: Add
      input:
        ret2882.1: {}
        attention_mask0.1: {}
      output:
        attn_weights68.1: {}
    aten::max_330:
      type: Max
      input:
        attn_weights68.1: {}
        '497': {}
      output:
        input68.1: {}
    aten::softmax_2443:
      type: Softmax
      input:
        input68.1: {}
      output:
        '9132': {}
      attr:
        axis: -1
    aten::matmul_5604:
      type: Matmul
      input:
        '9132': {}
        ret2877.1: {}
      output:
        ret2885.1: {}
    aten::transpose_2991:
      type: Reorder
      input:
        ret2885.1: {}
      output:
        ret2886.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5606:
      type: Reshape
      input:
        ret2886.1: {}
        '8994': {}
        '8996': {}
      output:
        ret2887.1: {}
    aten::mul_880:
      type: Mul
      input:
        ret2887.1: {}
        '1042': {}
      output:
        ret2888.1: {}
      attr:
        algorithm: mul
    aten::linear_3677:
      type: InnerProduct
      input:
        ret2888.1: {}
        '1043': {}
        aten::linear_3677_bias: {}
      output:
        ret2891.1: {}
    aten::add_3310:
      type: Add
      input:
        ret2830.1: {}
        ret2891.1: {}
      output:
        ret2892.1: {}
    aten::pow_2992:
      type: Pow
      input:
        ret2892.1: {}
        aten::pow_2992_other: {}
      output:
        ret2893.1: {}
    aten::mean_1042:
      type: ReduceMean
      input:
        ret2893.1: {}
      output:
        ret2894.1: {}
    aten::add_77:
      type: Add
      input:
        ret2894.1: {}
        '485': {}
      output:
        ret2895.1: {}
    aten::rsqrt_5609:
      type: Rsqrt
      input:
        ret2895.1: {}
      output:
        ret2896.1: {}
    aten::mul_5608:
      type: Mul
      input:
        ret2892.1: {}
        ret2896.1: {}
      output:
        ret2897.1: {}
      attr:
        algorithm: mul
    aten::mul_882:
      type: Mul
      input:
        '1044': {}
        ret2897.1: {}
      output:
        ret2898.1: {}
      attr:
        algorithm: mul
    aten::mul_883:
      type: Mul
      input:
        ret2898.1: {}
        '1045': {}
      output:
        ret2899.1: {}
      attr:
        algorithm: mul
    aten::linear_3678:
      type: InnerProduct
      input:
        ret2899.1: {}
        '1046': {}
        aten::linear_3678_bias: {}
      output:
        ret2902.1: {}
    aten::silu_5611:
      type: Swish
      input:
        ret2902.1: {}
      output:
        ret2903.1: {}
    aten::mul_885:
      type: Mul
      input:
        ret2898.1: {}
        '1047': {}
      output:
        ret2904.1: {}
      attr:
        algorithm: mul
    aten::linear_3679:
      type: InnerProduct
      input:
        ret2904.1: {}
        '1048': {}
        aten::linear_3679_bias: {}
      output:
        ret2907.1: {}
    aten::mul_5612:
      type: Mul
      input:
        ret2903.1: {}
        ret2907.1: {}
      output:
        ret2908.1: {}
      attr:
        algorithm: mul
    aten::mul_887:
      type: Mul
      input:
        ret2908.1: {}
        '1049': {}
      output:
        ret2909.1: {}
      attr:
        algorithm: mul
    aten::linear_3680:
      type: InnerProduct
      input:
        ret2909.1: {}
        '1050': {}
        aten::linear_3680_bias: {}
      output:
        ret2912.1: {}
    aten::add_3311:
      type: Add
      input:
        ret2892.1: {}
        ret2912.1: {}
      output:
        ret2913.1: {}
    aten::pow_2993:
      type: Pow
      input:
        ret2913.1: {}
        aten::pow_2993_other: {}
      output:
        ret2914.1: {}
    aten::mean_1043:
      type: ReduceMean
      input:
        ret2914.1: {}
      output:
        ret2915.1: {}
    aten::add_78:
      type: Add
      input:
        ret2915.1: {}
        '485': {}
      output:
        ret2916.1: {}
    aten::rsqrt_5616:
      type: Rsqrt
      input:
        ret2916.1: {}
      output:
        ret2917.1: {}
    aten::mul_5615:
      type: Mul
      input:
        ret2913.1: {}
        ret2917.1: {}
      output:
        ret2918.1: {}
      attr:
        algorithm: mul
    aten::mul_889:
      type: Mul
      input:
        '1051': {}
        ret2918.1: {}
      output:
        ret2919.1: {}
      attr:
        algorithm: mul
    aten::size_3424:
      type: Shape
      input:
        ret2919.1: {}
      output:
        '9217': {}
      attr:
        start: 0
        end: 0
    aten::size_3312:
      type: Shape
      input:
        ret2919.1: {}
      output:
        '9219': {}
      attr:
        start: 1
        end: 1
    aten::mul_890:
      type: Mul
      input:
        ret2919.1: {}
        '1052': {}
      output:
        ret2922.1: {}
      attr:
        algorithm: mul
    aten::linear_3681:
      type: InnerProduct
      input:
        ret2922.1: {}
        '1053': {}
        aten::linear_3681_bias: {}
      output:
        ret2925.1: {}
    aten::view_5618:
      type: View
      input:
        ret2925.1: {}
        '9217': {}
        '9219': {}
      output:
        ret2926.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2994:
      type: Reorder
      input:
        ret2926.1: {}
      output:
        ret2927.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_892:
      type: Mul
      input:
        ret2919.1: {}
        '1054': {}
      output:
        ret2928.1: {}
      attr:
        algorithm: mul
    aten::linear_3682:
      type: InnerProduct
      input:
        ret2928.1: {}
        '1055': {}
        aten::linear_3682_bias: {}
      output:
        ret2931.1: {}
    aten::view_5619:
      type: View
      input:
        ret2931.1: {}
        '9217': {}
        '9219': {}
      output:
        ret2932.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2995:
      type: Reorder
      input:
        ret2932.1: {}
      output:
        ret2933.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_894:
      type: Mul
      input:
        ret2919.1: {}
        '1056': {}
      output:
        ret2934.1: {}
      attr:
        algorithm: mul
    aten::linear_3683:
      type: InnerProduct
      input:
        ret2934.1: {}
        '1057': {}
        aten::linear_3683_bias: {}
      output:
        ret2937.1: {}
    aten::view_5620:
      type: View
      input:
        ret2937.1: {}
        '9217': {}
        '9219': {}
      output:
        ret2938.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_2996:
      type: Reorder
      input:
        ret2938.1: {}
      output:
        ret2939.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_2997:
      type: Shape
      input:
        ret2933.1: {}
      output:
        '9265': {}
      attr:
        start: 2
        end: 2
    aten::size_2998:
      type: Shape
      input:
        x69.1: {}
      output:
        '9268': {}
      attr:
        start: 2
        end: 2
    aten::add_3382:
      type: Add
      input:
        '9265': {}
        '9268': {}
      output:
        seq_len34.1: {}
    aten::slice_131:
      type: Slice
      input:
        '493': {}
        seq_len34.1: {}
      output:
        '9275': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_171:
      type: Slice
      input:
        '494': {}
        seq_len34.1: {}
      output:
        '9277': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_2999:
      type: Shape
      input:
        ret2927.1: {}
      output:
        '9279': {}
      attr:
        start: 2
        end: 2
    aten::add_3313:
      type: Add
      input:
        '9279': {}
        '9268': {}
      output:
        '9284': {}
    aten::slice_3000:
      type: Slice
      input:
        '9275': {}
        '9268': {}
        '9284': {}
      output:
        '9286': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2360:
      type: Slice
      input:
        '9286': {}
      output:
        '9287': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_3001:
      type: Slice
      input:
        '9277': {}
        '9268': {}
        '9284': {}
      output:
        '9288': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2361:
      type: Slice
      input:
        '9288': {}
      output:
        '9289': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5621:
      type: Mul
      input:
        ret2927.1: {}
        '9287': {}
      output:
        ret2943.1: {}
      attr:
        algorithm: mul
    aten::size_2362:
      type: Shape
      input:
        ret2927.1: {}
      output:
        '9293': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_246:
      type: Div
      input:
        '9293': {}
        '495': {}
      output:
        ret2945.1: {}
      attr:
        algorithm: div
    aten::slice_2363:
      type: Slice
      input:
        ret2927.1: {}
        ret2945.1: {}
      output:
        ret2946.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2364:
      type: Slice
      input:
        ret2927.1: {}
        ret2945.1: {}
      output:
        ret2947.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5628:
      type: Neg
      input:
        ret2947.1: {}
        aten::neg_5628_mul_val: {}
      output:
        ret2948.1: {}
      attr:
        algorithm: mul
    aten::cat_2523:
      type: Concat
      input:
        ret2948.1: {}
        ret2946.1: {}
      output:
        ret2949.1: {}
      attr:
        axis: -1
    aten::mul_5625:
      type: Mul
      input:
        ret2949.1: {}
        '9289': {}
      output:
        ret2950.1: {}
      attr:
        algorithm: mul
    aten::add_3314:
      type: Add
      input:
        ret2943.1: {}
        ret2950.1: {}
      output:
        args139.1: {}
    aten::mul_5623:
      type: Mul
      input:
        ret2933.1: {}
        '9287': {}
      output:
        ret2951.1: {}
      attr:
        algorithm: mul
    aten::size_2365:
      type: Shape
      input:
        ret2933.1: {}
      output:
        '9315': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_247:
      type: Div
      input:
        '9315': {}
        '495': {}
      output:
        ret2953.1: {}
      attr:
        algorithm: div
    aten::slice_2366:
      type: Slice
      input:
        ret2933.1: {}
        ret2953.1: {}
      output:
        ret2954.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2367:
      type: Slice
      input:
        ret2933.1: {}
        ret2953.1: {}
      output:
        ret2955.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5630:
      type: Neg
      input:
        ret2955.1: {}
        aten::neg_5630_mul_val: {}
      output:
        ret2956.1: {}
      attr:
        algorithm: mul
    aten::cat_2524:
      type: Concat
      input:
        ret2956.1: {}
        ret2954.1: {}
      output:
        ret2957.1: {}
      attr:
        axis: -1
    aten::mul_5626:
      type: Mul
      input:
        ret2957.1: {}
        '9289': {}
      output:
        ret2958.1: {}
      attr:
        algorithm: mul
    aten::add_3315:
      type: Add
      input:
        ret2951.1: {}
        ret2958.1: {}
      output:
        '9333': {}
    aten::cat_3002:
      type: Concat
      input:
        x69.1: {}
        '9333': {}
      output:
        ret2959.1: {}
      attr:
        axis: 2
    aten::cat_3003:
      type: Concat
      input:
        x70.1: {}
        ret2939.1: {}
      output:
        ret2960.1: {}
      attr:
        axis: 2
    aten::transpose_2368:
      type: Reorder
      input:
        ret2959.1: {}
      output:
        ret2961.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5633:
      type: Matmul
      input:
        args139.1: {}
        ret2961.1: {}
      output:
        ret2964.1: {}
    aten::div_291:
      type: Div
      input:
        ret2964.1: {}
        '496': {}
      output:
        ret2965.1: {}
      attr:
        algorithm: div
    aten::add_3316:
      type: Add
      input:
        ret2965.1: {}
        attention_mask0.1: {}
      output:
        attn_weights70.1: {}
    aten::max_331:
      type: Max
      input:
        attn_weights70.1: {}
        '497': {}
      output:
        input70.1: {}
    aten::softmax_2444:
      type: Softmax
      input:
        input70.1: {}
      output:
        '9355': {}
      attr:
        axis: -1
    aten::matmul_5636:
      type: Matmul
      input:
        '9355': {}
        ret2960.1: {}
      output:
        ret2968.1: {}
    aten::transpose_3004:
      type: Reorder
      input:
        ret2968.1: {}
      output:
        ret2969.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5638:
      type: Reshape
      input:
        ret2969.1: {}
        '9217': {}
        '9219': {}
      output:
        ret2970.1: {}
    aten::mul_896:
      type: Mul
      input:
        ret2970.1: {}
        '1058': {}
      output:
        ret2971.1: {}
      attr:
        algorithm: mul
    aten::linear_3684:
      type: InnerProduct
      input:
        ret2971.1: {}
        '1059': {}
        aten::linear_3684_bias: {}
      output:
        ret2974.1: {}
    aten::add_3317:
      type: Add
      input:
        ret2913.1: {}
        ret2974.1: {}
      output:
        ret2975.1: {}
    aten::pow_3005:
      type: Pow
      input:
        ret2975.1: {}
        aten::pow_3005_other: {}
      output:
        ret2976.1: {}
    aten::mean_1044:
      type: ReduceMean
      input:
        ret2976.1: {}
      output:
        ret2977.1: {}
    aten::add_79:
      type: Add
      input:
        ret2977.1: {}
        '485': {}
      output:
        ret2978.1: {}
    aten::rsqrt_5641:
      type: Rsqrt
      input:
        ret2978.1: {}
      output:
        ret2979.1: {}
    aten::mul_5640:
      type: Mul
      input:
        ret2975.1: {}
        ret2979.1: {}
      output:
        ret2980.1: {}
      attr:
        algorithm: mul
    aten::mul_898:
      type: Mul
      input:
        '1060': {}
        ret2980.1: {}
      output:
        ret2981.1: {}
      attr:
        algorithm: mul
    aten::mul_899:
      type: Mul
      input:
        ret2981.1: {}
        '1061': {}
      output:
        ret2982.1: {}
      attr:
        algorithm: mul
    aten::linear_3685:
      type: InnerProduct
      input:
        ret2982.1: {}
        '1062': {}
        aten::linear_3685_bias: {}
      output:
        ret2985.1: {}
    aten::silu_5643:
      type: Swish
      input:
        ret2985.1: {}
      output:
        ret2986.1: {}
    aten::mul_901:
      type: Mul
      input:
        ret2981.1: {}
        '1063': {}
      output:
        ret2987.1: {}
      attr:
        algorithm: mul
    aten::linear_3686:
      type: InnerProduct
      input:
        ret2987.1: {}
        '1064': {}
        aten::linear_3686_bias: {}
      output:
        ret2990.1: {}
    aten::mul_5644:
      type: Mul
      input:
        ret2986.1: {}
        ret2990.1: {}
      output:
        ret2991.1: {}
      attr:
        algorithm: mul
    aten::mul_903:
      type: Mul
      input:
        ret2991.1: {}
        '1065': {}
      output:
        ret2992.1: {}
      attr:
        algorithm: mul
    aten::linear_3687:
      type: InnerProduct
      input:
        ret2992.1: {}
        '1066': {}
        aten::linear_3687_bias: {}
      output:
        ret2995.1: {}
    aten::add_3318:
      type: Add
      input:
        ret2975.1: {}
        ret2995.1: {}
      output:
        ret2996.1: {}
    aten::pow_3006:
      type: Pow
      input:
        ret2996.1: {}
        aten::pow_3006_other: {}
      output:
        ret2997.1: {}
    aten::mean_1045:
      type: ReduceMean
      input:
        ret2997.1: {}
      output:
        ret2998.1: {}
    aten::add_80:
      type: Add
      input:
        ret2998.1: {}
        '485': {}
      output:
        ret2999.1: {}
    aten::rsqrt_5648:
      type: Rsqrt
      input:
        ret2999.1: {}
      output:
        ret3000.1: {}
    aten::mul_5647:
      type: Mul
      input:
        ret2996.1: {}
        ret3000.1: {}
      output:
        ret3001.1: {}
      attr:
        algorithm: mul
    aten::mul_905:
      type: Mul
      input:
        '1067': {}
        ret3001.1: {}
      output:
        ret3002.1: {}
      attr:
        algorithm: mul
    aten::size_3425:
      type: Shape
      input:
        ret3002.1: {}
      output:
        '9440': {}
      attr:
        start: 0
        end: 0
    aten::size_3319:
      type: Shape
      input:
        ret3002.1: {}
      output:
        '9442': {}
      attr:
        start: 1
        end: 1
    aten::mul_906:
      type: Mul
      input:
        ret3002.1: {}
        '1068': {}
      output:
        ret3005.1: {}
      attr:
        algorithm: mul
    aten::linear_3688:
      type: InnerProduct
      input:
        ret3005.1: {}
        '1069': {}
        aten::linear_3688_bias: {}
      output:
        ret3008.1: {}
    aten::view_5650:
      type: View
      input:
        ret3008.1: {}
        '9440': {}
        '9442': {}
      output:
        ret3009.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3007:
      type: Reorder
      input:
        ret3009.1: {}
      output:
        ret3010.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_908:
      type: Mul
      input:
        ret3002.1: {}
        '1070': {}
      output:
        ret3011.1: {}
      attr:
        algorithm: mul
    aten::linear_3689:
      type: InnerProduct
      input:
        ret3011.1: {}
        '1071': {}
        aten::linear_3689_bias: {}
      output:
        ret3014.1: {}
    aten::view_5651:
      type: View
      input:
        ret3014.1: {}
        '9440': {}
        '9442': {}
      output:
        ret3015.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3008:
      type: Reorder
      input:
        ret3015.1: {}
      output:
        ret3016.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_910:
      type: Mul
      input:
        ret3002.1: {}
        '1072': {}
      output:
        ret3017.1: {}
      attr:
        algorithm: mul
    aten::linear_3690:
      type: InnerProduct
      input:
        ret3017.1: {}
        '1073': {}
        aten::linear_3690_bias: {}
      output:
        ret3020.1: {}
    aten::view_5652:
      type: View
      input:
        ret3020.1: {}
        '9440': {}
        '9442': {}
      output:
        ret3021.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3009:
      type: Reorder
      input:
        ret3021.1: {}
      output:
        ret3022.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_3010:
      type: Shape
      input:
        ret3016.1: {}
      output:
        '9488': {}
      attr:
        start: 2
        end: 2
    aten::size_3011:
      type: Shape
      input:
        x71.1: {}
      output:
        '9491': {}
      attr:
        start: 2
        end: 2
    aten::add_3383:
      type: Add
      input:
        '9488': {}
        '9491': {}
      output:
        seq_len35.1: {}
    aten::slice_132:
      type: Slice
      input:
        '493': {}
        seq_len35.1: {}
      output:
        '9498': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_172:
      type: Slice
      input:
        '494': {}
        seq_len35.1: {}
      output:
        '9500': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_3012:
      type: Shape
      input:
        ret3010.1: {}
      output:
        '9502': {}
      attr:
        start: 2
        end: 2
    aten::add_3320:
      type: Add
      input:
        '9502': {}
        '9491': {}
      output:
        '9507': {}
    aten::slice_3013:
      type: Slice
      input:
        '9498': {}
        '9491': {}
        '9507': {}
      output:
        '9509': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2369:
      type: Slice
      input:
        '9509': {}
      output:
        '9510': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_3014:
      type: Slice
      input:
        '9500': {}
        '9491': {}
        '9507': {}
      output:
        '9511': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2370:
      type: Slice
      input:
        '9511': {}
      output:
        '9512': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5653:
      type: Mul
      input:
        ret3010.1: {}
        '9510': {}
      output:
        ret3026.1: {}
      attr:
        algorithm: mul
    aten::size_2371:
      type: Shape
      input:
        ret3010.1: {}
      output:
        '9516': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_248:
      type: Div
      input:
        '9516': {}
        '495': {}
      output:
        ret3028.1: {}
      attr:
        algorithm: div
    aten::slice_2372:
      type: Slice
      input:
        ret3010.1: {}
        ret3028.1: {}
      output:
        ret3029.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2373:
      type: Slice
      input:
        ret3010.1: {}
        ret3028.1: {}
      output:
        ret3030.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5660:
      type: Neg
      input:
        ret3030.1: {}
        aten::neg_5660_mul_val: {}
      output:
        ret3031.1: {}
      attr:
        algorithm: mul
    aten::cat_2525:
      type: Concat
      input:
        ret3031.1: {}
        ret3029.1: {}
      output:
        ret3032.1: {}
      attr:
        axis: -1
    aten::mul_5657:
      type: Mul
      input:
        ret3032.1: {}
        '9512': {}
      output:
        ret3033.1: {}
      attr:
        algorithm: mul
    aten::add_3321:
      type: Add
      input:
        ret3026.1: {}
        ret3033.1: {}
      output:
        args143.1: {}
    aten::mul_5655:
      type: Mul
      input:
        ret3016.1: {}
        '9510': {}
      output:
        ret3034.1: {}
      attr:
        algorithm: mul
    aten::size_2374:
      type: Shape
      input:
        ret3016.1: {}
      output:
        '9538': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_249:
      type: Div
      input:
        '9538': {}
        '495': {}
      output:
        ret3036.1: {}
      attr:
        algorithm: div
    aten::slice_2375:
      type: Slice
      input:
        ret3016.1: {}
        ret3036.1: {}
      output:
        ret3037.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2376:
      type: Slice
      input:
        ret3016.1: {}
        ret3036.1: {}
      output:
        ret3038.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5662:
      type: Neg
      input:
        ret3038.1: {}
        aten::neg_5662_mul_val: {}
      output:
        ret3039.1: {}
      attr:
        algorithm: mul
    aten::cat_2526:
      type: Concat
      input:
        ret3039.1: {}
        ret3037.1: {}
      output:
        ret3040.1: {}
      attr:
        axis: -1
    aten::mul_5658:
      type: Mul
      input:
        ret3040.1: {}
        '9512': {}
      output:
        ret3041.1: {}
      attr:
        algorithm: mul
    aten::add_3322:
      type: Add
      input:
        ret3034.1: {}
        ret3041.1: {}
      output:
        '9556': {}
    aten::cat_3015:
      type: Concat
      input:
        x71.1: {}
        '9556': {}
      output:
        ret3042.1: {}
      attr:
        axis: 2
    aten::cat_3016:
      type: Concat
      input:
        x72.1: {}
        ret3022.1: {}
      output:
        ret3043.1: {}
      attr:
        axis: 2
    aten::transpose_2377:
      type: Reorder
      input:
        ret3042.1: {}
      output:
        ret3044.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5665:
      type: Matmul
      input:
        args143.1: {}
        ret3044.1: {}
      output:
        ret3047.1: {}
    aten::div_292:
      type: Div
      input:
        ret3047.1: {}
        '496': {}
      output:
        ret3048.1: {}
      attr:
        algorithm: div
    aten::add_3323:
      type: Add
      input:
        ret3048.1: {}
        attention_mask0.1: {}
      output:
        attn_weights72.1: {}
    aten::max_332:
      type: Max
      input:
        attn_weights72.1: {}
        '497': {}
      output:
        input72.1: {}
    aten::softmax_2445:
      type: Softmax
      input:
        input72.1: {}
      output:
        '9578': {}
      attr:
        axis: -1
    aten::matmul_5668:
      type: Matmul
      input:
        '9578': {}
        ret3043.1: {}
      output:
        ret3051.1: {}
    aten::transpose_3017:
      type: Reorder
      input:
        ret3051.1: {}
      output:
        ret3052.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5670:
      type: Reshape
      input:
        ret3052.1: {}
        '9440': {}
        '9442': {}
      output:
        ret3053.1: {}
    aten::mul_912:
      type: Mul
      input:
        ret3053.1: {}
        '1074': {}
      output:
        ret3054.1: {}
      attr:
        algorithm: mul
    aten::linear_3691:
      type: InnerProduct
      input:
        ret3054.1: {}
        '1075': {}
        aten::linear_3691_bias: {}
      output:
        ret3057.1: {}
    aten::add_3324:
      type: Add
      input:
        ret2996.1: {}
        ret3057.1: {}
      output:
        ret3058.1: {}
    aten::pow_3018:
      type: Pow
      input:
        ret3058.1: {}
        aten::pow_3018_other: {}
      output:
        ret3059.1: {}
    aten::mean_1046:
      type: ReduceMean
      input:
        ret3059.1: {}
      output:
        ret3060.1: {}
    aten::add_81:
      type: Add
      input:
        ret3060.1: {}
        '485': {}
      output:
        ret3061.1: {}
    aten::rsqrt_5673:
      type: Rsqrt
      input:
        ret3061.1: {}
      output:
        ret3062.1: {}
    aten::mul_5672:
      type: Mul
      input:
        ret3058.1: {}
        ret3062.1: {}
      output:
        ret3063.1: {}
      attr:
        algorithm: mul
    aten::mul_914:
      type: Mul
      input:
        '1076': {}
        ret3063.1: {}
      output:
        ret3064.1: {}
      attr:
        algorithm: mul
    aten::mul_915:
      type: Mul
      input:
        ret3064.1: {}
        '1077': {}
      output:
        ret3065.1: {}
      attr:
        algorithm: mul
    aten::linear_3692:
      type: InnerProduct
      input:
        ret3065.1: {}
        '1078': {}
        aten::linear_3692_bias: {}
      output:
        ret3068.1: {}
    aten::silu_5675:
      type: Swish
      input:
        ret3068.1: {}
      output:
        ret3069.1: {}
    aten::mul_917:
      type: Mul
      input:
        ret3064.1: {}
        '1079': {}
      output:
        ret3070.1: {}
      attr:
        algorithm: mul
    aten::linear_3693:
      type: InnerProduct
      input:
        ret3070.1: {}
        '1080': {}
        aten::linear_3693_bias: {}
      output:
        ret3073.1: {}
    aten::mul_5676:
      type: Mul
      input:
        ret3069.1: {}
        ret3073.1: {}
      output:
        ret3074.1: {}
      attr:
        algorithm: mul
    aten::mul_919:
      type: Mul
      input:
        ret3074.1: {}
        '1081': {}
      output:
        ret3075.1: {}
      attr:
        algorithm: mul
    aten::linear_3694:
      type: InnerProduct
      input:
        ret3075.1: {}
        '1082': {}
        aten::linear_3694_bias: {}
      output:
        ret3078.1: {}
    aten::add_3325:
      type: Add
      input:
        ret3058.1: {}
        ret3078.1: {}
      output:
        ret3079.1: {}
    aten::pow_3019:
      type: Pow
      input:
        ret3079.1: {}
        aten::pow_3019_other: {}
      output:
        ret3080.1: {}
    aten::mean_1047:
      type: ReduceMean
      input:
        ret3080.1: {}
      output:
        ret3081.1: {}
    aten::add_82:
      type: Add
      input:
        ret3081.1: {}
        '485': {}
      output:
        ret3082.1: {}
    aten::rsqrt_5680:
      type: Rsqrt
      input:
        ret3082.1: {}
      output:
        ret3083.1: {}
    aten::mul_5679:
      type: Mul
      input:
        ret3079.1: {}
        ret3083.1: {}
      output:
        ret3084.1: {}
      attr:
        algorithm: mul
    aten::mul_921:
      type: Mul
      input:
        '1083': {}
        ret3084.1: {}
      output:
        ret3085.1: {}
      attr:
        algorithm: mul
    aten::size_3426:
      type: Shape
      input:
        ret3085.1: {}
      output:
        '9663': {}
      attr:
        start: 0
        end: 0
    aten::size_3326:
      type: Shape
      input:
        ret3085.1: {}
      output:
        '9665': {}
      attr:
        start: 1
        end: 1
    aten::mul_922:
      type: Mul
      input:
        ret3085.1: {}
        '1084': {}
      output:
        ret3088.1: {}
      attr:
        algorithm: mul
    aten::linear_3695:
      type: InnerProduct
      input:
        ret3088.1: {}
        '1085': {}
        aten::linear_3695_bias: {}
      output:
        ret3091.1: {}
    aten::view_5682:
      type: View
      input:
        ret3091.1: {}
        '9663': {}
        '9665': {}
      output:
        ret3092.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3020:
      type: Reorder
      input:
        ret3092.1: {}
      output:
        ret3093.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_924:
      type: Mul
      input:
        ret3085.1: {}
        '1086': {}
      output:
        ret3094.1: {}
      attr:
        algorithm: mul
    aten::linear_3696:
      type: InnerProduct
      input:
        ret3094.1: {}
        '1087': {}
        aten::linear_3696_bias: {}
      output:
        ret3097.1: {}
    aten::view_5683:
      type: View
      input:
        ret3097.1: {}
        '9663': {}
        '9665': {}
      output:
        ret3098.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3021:
      type: Reorder
      input:
        ret3098.1: {}
      output:
        ret3099.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_926:
      type: Mul
      input:
        ret3085.1: {}
        '1088': {}
      output:
        ret3100.1: {}
      attr:
        algorithm: mul
    aten::linear_3697:
      type: InnerProduct
      input:
        ret3100.1: {}
        '1089': {}
        aten::linear_3697_bias: {}
      output:
        ret3103.1: {}
    aten::view_5684:
      type: View
      input:
        ret3103.1: {}
        '9663': {}
        '9665': {}
      output:
        ret3104.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3022:
      type: Reorder
      input:
        ret3104.1: {}
      output:
        ret3105.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_3023:
      type: Shape
      input:
        ret3099.1: {}
      output:
        '9711': {}
      attr:
        start: 2
        end: 2
    aten::size_3024:
      type: Shape
      input:
        x73.1: {}
      output:
        '9714': {}
      attr:
        start: 2
        end: 2
    aten::add_3384:
      type: Add
      input:
        '9711': {}
        '9714': {}
      output:
        seq_len36.1: {}
    aten::slice_133:
      type: Slice
      input:
        '493': {}
        seq_len36.1: {}
      output:
        '9721': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_173:
      type: Slice
      input:
        '494': {}
        seq_len36.1: {}
      output:
        '9723': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_3025:
      type: Shape
      input:
        ret3093.1: {}
      output:
        '9725': {}
      attr:
        start: 2
        end: 2
    aten::add_3327:
      type: Add
      input:
        '9725': {}
        '9714': {}
      output:
        '9730': {}
    aten::slice_3026:
      type: Slice
      input:
        '9721': {}
        '9714': {}
        '9730': {}
      output:
        '9732': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2378:
      type: Slice
      input:
        '9732': {}
      output:
        '9733': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_3027:
      type: Slice
      input:
        '9723': {}
        '9714': {}
        '9730': {}
      output:
        '9734': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2379:
      type: Slice
      input:
        '9734': {}
      output:
        '9735': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5685:
      type: Mul
      input:
        ret3093.1: {}
        '9733': {}
      output:
        ret3109.1: {}
      attr:
        algorithm: mul
    aten::size_2380:
      type: Shape
      input:
        ret3093.1: {}
      output:
        '9739': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_250:
      type: Div
      input:
        '9739': {}
        '495': {}
      output:
        ret3111.1: {}
      attr:
        algorithm: div
    aten::slice_2381:
      type: Slice
      input:
        ret3093.1: {}
        ret3111.1: {}
      output:
        ret3112.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2382:
      type: Slice
      input:
        ret3093.1: {}
        ret3111.1: {}
      output:
        ret3113.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5692:
      type: Neg
      input:
        ret3113.1: {}
        aten::neg_5692_mul_val: {}
      output:
        ret3114.1: {}
      attr:
        algorithm: mul
    aten::cat_2527:
      type: Concat
      input:
        ret3114.1: {}
        ret3112.1: {}
      output:
        ret3115.1: {}
      attr:
        axis: -1
    aten::mul_5689:
      type: Mul
      input:
        ret3115.1: {}
        '9735': {}
      output:
        ret3116.1: {}
      attr:
        algorithm: mul
    aten::add_3328:
      type: Add
      input:
        ret3109.1: {}
        ret3116.1: {}
      output:
        args147.1: {}
    aten::mul_5687:
      type: Mul
      input:
        ret3099.1: {}
        '9733': {}
      output:
        ret3117.1: {}
      attr:
        algorithm: mul
    aten::size_2383:
      type: Shape
      input:
        ret3099.1: {}
      output:
        '9761': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_251:
      type: Div
      input:
        '9761': {}
        '495': {}
      output:
        ret3119.1: {}
      attr:
        algorithm: div
    aten::slice_2384:
      type: Slice
      input:
        ret3099.1: {}
        ret3119.1: {}
      output:
        ret3120.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2385:
      type: Slice
      input:
        ret3099.1: {}
        ret3119.1: {}
      output:
        ret3121.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5694:
      type: Neg
      input:
        ret3121.1: {}
        aten::neg_5694_mul_val: {}
      output:
        ret3122.1: {}
      attr:
        algorithm: mul
    aten::cat_2528:
      type: Concat
      input:
        ret3122.1: {}
        ret3120.1: {}
      output:
        ret3123.1: {}
      attr:
        axis: -1
    aten::mul_5690:
      type: Mul
      input:
        ret3123.1: {}
        '9735': {}
      output:
        ret3124.1: {}
      attr:
        algorithm: mul
    aten::add_3329:
      type: Add
      input:
        ret3117.1: {}
        ret3124.1: {}
      output:
        '9779': {}
    aten::cat_3028:
      type: Concat
      input:
        x73.1: {}
        '9779': {}
      output:
        ret3125.1: {}
      attr:
        axis: 2
    aten::cat_3029:
      type: Concat
      input:
        x74.1: {}
        ret3105.1: {}
      output:
        ret3126.1: {}
      attr:
        axis: 2
    aten::transpose_2386:
      type: Reorder
      input:
        ret3125.1: {}
      output:
        ret3127.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5697:
      type: Matmul
      input:
        args147.1: {}
        ret3127.1: {}
      output:
        ret3130.1: {}
    aten::div_293:
      type: Div
      input:
        ret3130.1: {}
        '496': {}
      output:
        ret3131.1: {}
      attr:
        algorithm: div
    aten::add_3330:
      type: Add
      input:
        ret3131.1: {}
        attention_mask0.1: {}
      output:
        attn_weights74.1: {}
    aten::max_333:
      type: Max
      input:
        attn_weights74.1: {}
        '497': {}
      output:
        input74.1: {}
    aten::softmax_2446:
      type: Softmax
      input:
        input74.1: {}
      output:
        '9801': {}
      attr:
        axis: -1
    aten::matmul_5700:
      type: Matmul
      input:
        '9801': {}
        ret3126.1: {}
      output:
        ret3134.1: {}
    aten::transpose_3030:
      type: Reorder
      input:
        ret3134.1: {}
      output:
        ret3135.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5702:
      type: Reshape
      input:
        ret3135.1: {}
        '9663': {}
        '9665': {}
      output:
        ret3136.1: {}
    aten::mul_928:
      type: Mul
      input:
        ret3136.1: {}
        '1090': {}
      output:
        ret3137.1: {}
      attr:
        algorithm: mul
    aten::linear_3698:
      type: InnerProduct
      input:
        ret3137.1: {}
        '1091': {}
        aten::linear_3698_bias: {}
      output:
        ret3140.1: {}
    aten::add_3331:
      type: Add
      input:
        ret3079.1: {}
        ret3140.1: {}
      output:
        ret3141.1: {}
    aten::pow_3031:
      type: Pow
      input:
        ret3141.1: {}
        aten::pow_3031_other: {}
      output:
        ret3142.1: {}
    aten::mean_1048:
      type: ReduceMean
      input:
        ret3142.1: {}
      output:
        ret3143.1: {}
    aten::add_83:
      type: Add
      input:
        ret3143.1: {}
        '485': {}
      output:
        ret3144.1: {}
    aten::rsqrt_5705:
      type: Rsqrt
      input:
        ret3144.1: {}
      output:
        ret3145.1: {}
    aten::mul_5704:
      type: Mul
      input:
        ret3141.1: {}
        ret3145.1: {}
      output:
        ret3146.1: {}
      attr:
        algorithm: mul
    aten::mul_930:
      type: Mul
      input:
        '1092': {}
        ret3146.1: {}
      output:
        ret3147.1: {}
      attr:
        algorithm: mul
    aten::mul_931:
      type: Mul
      input:
        ret3147.1: {}
        '1093': {}
      output:
        ret3148.1: {}
      attr:
        algorithm: mul
    aten::linear_3699:
      type: InnerProduct
      input:
        ret3148.1: {}
        '1094': {}
        aten::linear_3699_bias: {}
      output:
        ret3151.1: {}
    aten::silu_5707:
      type: Swish
      input:
        ret3151.1: {}
      output:
        ret3152.1: {}
    aten::mul_933:
      type: Mul
      input:
        ret3147.1: {}
        '1095': {}
      output:
        ret3153.1: {}
      attr:
        algorithm: mul
    aten::linear_3700:
      type: InnerProduct
      input:
        ret3153.1: {}
        '1096': {}
        aten::linear_3700_bias: {}
      output:
        ret3156.1: {}
    aten::mul_5708:
      type: Mul
      input:
        ret3152.1: {}
        ret3156.1: {}
      output:
        ret3157.1: {}
      attr:
        algorithm: mul
    aten::mul_935:
      type: Mul
      input:
        ret3157.1: {}
        '1097': {}
      output:
        ret3158.1: {}
      attr:
        algorithm: mul
    aten::linear_3701:
      type: InnerProduct
      input:
        ret3158.1: {}
        '1098': {}
        aten::linear_3701_bias: {}
      output:
        ret3161.1: {}
    aten::add_3332:
      type: Add
      input:
        ret3141.1: {}
        ret3161.1: {}
      output:
        ret3162.1: {}
    aten::pow_3032:
      type: Pow
      input:
        ret3162.1: {}
        aten::pow_3032_other: {}
      output:
        ret3163.1: {}
    aten::mean_1049:
      type: ReduceMean
      input:
        ret3163.1: {}
      output:
        ret3164.1: {}
    aten::add_84:
      type: Add
      input:
        ret3164.1: {}
        '485': {}
      output:
        ret3165.1: {}
    aten::rsqrt_5712:
      type: Rsqrt
      input:
        ret3165.1: {}
      output:
        ret3166.1: {}
    aten::mul_5711:
      type: Mul
      input:
        ret3162.1: {}
        ret3166.1: {}
      output:
        ret3167.1: {}
      attr:
        algorithm: mul
    aten::mul_937:
      type: Mul
      input:
        '1099': {}
        ret3167.1: {}
      output:
        ret3168.1: {}
      attr:
        algorithm: mul
    aten::size_3427:
      type: Shape
      input:
        ret3168.1: {}
      output:
        '9886': {}
      attr:
        start: 0
        end: 0
    aten::size_3333:
      type: Shape
      input:
        ret3168.1: {}
      output:
        '9888': {}
      attr:
        start: 1
        end: 1
    aten::mul_938:
      type: Mul
      input:
        ret3168.1: {}
        '1100': {}
      output:
        ret3171.1: {}
      attr:
        algorithm: mul
    aten::linear_3702:
      type: InnerProduct
      input:
        ret3171.1: {}
        '1101': {}
        aten::linear_3702_bias: {}
      output:
        ret3174.1: {}
    aten::view_5714:
      type: View
      input:
        ret3174.1: {}
        '9886': {}
        '9888': {}
      output:
        ret3175.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3033:
      type: Reorder
      input:
        ret3175.1: {}
      output:
        ret3176.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_940:
      type: Mul
      input:
        ret3168.1: {}
        '1102': {}
      output:
        ret3177.1: {}
      attr:
        algorithm: mul
    aten::linear_3703:
      type: InnerProduct
      input:
        ret3177.1: {}
        '1103': {}
        aten::linear_3703_bias: {}
      output:
        ret3180.1: {}
    aten::view_5715:
      type: View
      input:
        ret3180.1: {}
        '9886': {}
        '9888': {}
      output:
        ret3181.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3034:
      type: Reorder
      input:
        ret3181.1: {}
      output:
        ret3182.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_942:
      type: Mul
      input:
        ret3168.1: {}
        '1104': {}
      output:
        ret3183.1: {}
      attr:
        algorithm: mul
    aten::linear_3704:
      type: InnerProduct
      input:
        ret3183.1: {}
        '1105': {}
        aten::linear_3704_bias: {}
      output:
        ret3186.1: {}
    aten::view_5716:
      type: View
      input:
        ret3186.1: {}
        '9886': {}
        '9888': {}
      output:
        ret3187.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3035:
      type: Reorder
      input:
        ret3187.1: {}
      output:
        ret3188.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_3036:
      type: Shape
      input:
        ret3182.1: {}
      output:
        '9934': {}
      attr:
        start: 2
        end: 2
    aten::size_3037:
      type: Shape
      input:
        x75.1: {}
      output:
        '9937': {}
      attr:
        start: 2
        end: 2
    aten::add_3385:
      type: Add
      input:
        '9934': {}
        '9937': {}
      output:
        seq_len37.1: {}
    aten::slice_134:
      type: Slice
      input:
        '493': {}
        seq_len37.1: {}
      output:
        '9944': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_174:
      type: Slice
      input:
        '494': {}
        seq_len37.1: {}
      output:
        '9946': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_3038:
      type: Shape
      input:
        ret3176.1: {}
      output:
        '9948': {}
      attr:
        start: 2
        end: 2
    aten::add_3334:
      type: Add
      input:
        '9948': {}
        '9937': {}
      output:
        '9953': {}
    aten::slice_3039:
      type: Slice
      input:
        '9944': {}
        '9937': {}
        '9953': {}
      output:
        '9955': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2387:
      type: Slice
      input:
        '9955': {}
      output:
        '9956': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_3040:
      type: Slice
      input:
        '9946': {}
        '9937': {}
        '9953': {}
      output:
        '9957': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2388:
      type: Slice
      input:
        '9957': {}
      output:
        '9958': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5717:
      type: Mul
      input:
        ret3176.1: {}
        '9956': {}
      output:
        ret3192.1: {}
      attr:
        algorithm: mul
    aten::size_2389:
      type: Shape
      input:
        ret3176.1: {}
      output:
        '9962': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_252:
      type: Div
      input:
        '9962': {}
        '495': {}
      output:
        ret3194.1: {}
      attr:
        algorithm: div
    aten::slice_2390:
      type: Slice
      input:
        ret3176.1: {}
        ret3194.1: {}
      output:
        ret3195.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2391:
      type: Slice
      input:
        ret3176.1: {}
        ret3194.1: {}
      output:
        ret3196.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5724:
      type: Neg
      input:
        ret3196.1: {}
        aten::neg_5724_mul_val: {}
      output:
        ret3197.1: {}
      attr:
        algorithm: mul
    aten::cat_2529:
      type: Concat
      input:
        ret3197.1: {}
        ret3195.1: {}
      output:
        ret3198.1: {}
      attr:
        axis: -1
    aten::mul_5721:
      type: Mul
      input:
        ret3198.1: {}
        '9958': {}
      output:
        ret3199.1: {}
      attr:
        algorithm: mul
    aten::add_3335:
      type: Add
      input:
        ret3192.1: {}
        ret3199.1: {}
      output:
        args151.1: {}
    aten::mul_5719:
      type: Mul
      input:
        ret3182.1: {}
        '9956': {}
      output:
        ret3200.1: {}
      attr:
        algorithm: mul
    aten::size_2392:
      type: Shape
      input:
        ret3182.1: {}
      output:
        '9984': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_253:
      type: Div
      input:
        '9984': {}
        '495': {}
      output:
        ret3202.1: {}
      attr:
        algorithm: div
    aten::slice_2393:
      type: Slice
      input:
        ret3182.1: {}
        ret3202.1: {}
      output:
        ret3203.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2394:
      type: Slice
      input:
        ret3182.1: {}
        ret3202.1: {}
      output:
        ret3204.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5726:
      type: Neg
      input:
        ret3204.1: {}
        aten::neg_5726_mul_val: {}
      output:
        ret3205.1: {}
      attr:
        algorithm: mul
    aten::cat_2530:
      type: Concat
      input:
        ret3205.1: {}
        ret3203.1: {}
      output:
        ret3206.1: {}
      attr:
        axis: -1
    aten::mul_5722:
      type: Mul
      input:
        ret3206.1: {}
        '9958': {}
      output:
        ret3207.1: {}
      attr:
        algorithm: mul
    aten::add_3336:
      type: Add
      input:
        ret3200.1: {}
        ret3207.1: {}
      output:
        '10002': {}
    aten::cat_3041:
      type: Concat
      input:
        x75.1: {}
        '10002': {}
      output:
        ret3208.1: {}
      attr:
        axis: 2
    aten::cat_3042:
      type: Concat
      input:
        x76.1: {}
        ret3188.1: {}
      output:
        ret3209.1: {}
      attr:
        axis: 2
    aten::transpose_2395:
      type: Reorder
      input:
        ret3208.1: {}
      output:
        ret3210.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5729:
      type: Matmul
      input:
        args151.1: {}
        ret3210.1: {}
      output:
        ret3213.1: {}
    aten::div_294:
      type: Div
      input:
        ret3213.1: {}
        '496': {}
      output:
        ret3214.1: {}
      attr:
        algorithm: div
    aten::add_3337:
      type: Add
      input:
        ret3214.1: {}
        attention_mask0.1: {}
      output:
        attn_weights76.1: {}
    aten::max_334:
      type: Max
      input:
        attn_weights76.1: {}
        '497': {}
      output:
        input76.1: {}
    aten::softmax_2447:
      type: Softmax
      input:
        input76.1: {}
      output:
        '10024': {}
      attr:
        axis: -1
    aten::matmul_5732:
      type: Matmul
      input:
        '10024': {}
        ret3209.1: {}
      output:
        ret3217.1: {}
    aten::transpose_3043:
      type: Reorder
      input:
        ret3217.1: {}
      output:
        ret3218.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5734:
      type: Reshape
      input:
        ret3218.1: {}
        '9886': {}
        '9888': {}
      output:
        ret3219.1: {}
    aten::mul_944:
      type: Mul
      input:
        ret3219.1: {}
        '1106': {}
      output:
        ret3220.1: {}
      attr:
        algorithm: mul
    aten::linear_3705:
      type: InnerProduct
      input:
        ret3220.1: {}
        '1107': {}
        aten::linear_3705_bias: {}
      output:
        ret3223.1: {}
    aten::add_3338:
      type: Add
      input:
        ret3162.1: {}
        ret3223.1: {}
      output:
        ret3224.1: {}
    aten::pow_3044:
      type: Pow
      input:
        ret3224.1: {}
        aten::pow_3044_other: {}
      output:
        ret3225.1: {}
    aten::mean_1050:
      type: ReduceMean
      input:
        ret3225.1: {}
      output:
        ret3226.1: {}
    aten::add_85:
      type: Add
      input:
        ret3226.1: {}
        '485': {}
      output:
        ret3227.1: {}
    aten::rsqrt_5737:
      type: Rsqrt
      input:
        ret3227.1: {}
      output:
        ret3228.1: {}
    aten::mul_5736:
      type: Mul
      input:
        ret3224.1: {}
        ret3228.1: {}
      output:
        ret3229.1: {}
      attr:
        algorithm: mul
    aten::mul_946:
      type: Mul
      input:
        '1108': {}
        ret3229.1: {}
      output:
        ret3230.1: {}
      attr:
        algorithm: mul
    aten::mul_947:
      type: Mul
      input:
        ret3230.1: {}
        '1109': {}
      output:
        ret3231.1: {}
      attr:
        algorithm: mul
    aten::linear_3706:
      type: InnerProduct
      input:
        ret3231.1: {}
        '1110': {}
        aten::linear_3706_bias: {}
      output:
        ret3234.1: {}
    aten::silu_5739:
      type: Swish
      input:
        ret3234.1: {}
      output:
        ret3235.1: {}
    aten::mul_949:
      type: Mul
      input:
        ret3230.1: {}
        '1111': {}
      output:
        ret3236.1: {}
      attr:
        algorithm: mul
    aten::linear_3707:
      type: InnerProduct
      input:
        ret3236.1: {}
        '1112': {}
        aten::linear_3707_bias: {}
      output:
        ret3239.1: {}
    aten::mul_5740:
      type: Mul
      input:
        ret3235.1: {}
        ret3239.1: {}
      output:
        ret3240.1: {}
      attr:
        algorithm: mul
    aten::mul_951:
      type: Mul
      input:
        ret3240.1: {}
        '1113': {}
      output:
        ret3241.1: {}
      attr:
        algorithm: mul
    aten::linear_3708:
      type: InnerProduct
      input:
        ret3241.1: {}
        '1114': {}
        aten::linear_3708_bias: {}
      output:
        ret3244.1: {}
    aten::add_3339:
      type: Add
      input:
        ret3224.1: {}
        ret3244.1: {}
      output:
        ret3245.1: {}
    aten::pow_3045:
      type: Pow
      input:
        ret3245.1: {}
        aten::pow_3045_other: {}
      output:
        ret3246.1: {}
    aten::mean_1051:
      type: ReduceMean
      input:
        ret3246.1: {}
      output:
        ret3247.1: {}
    aten::add_86:
      type: Add
      input:
        ret3247.1: {}
        '485': {}
      output:
        ret3248.1: {}
    aten::rsqrt_5744:
      type: Rsqrt
      input:
        ret3248.1: {}
      output:
        ret3249.1: {}
    aten::mul_5743:
      type: Mul
      input:
        ret3245.1: {}
        ret3249.1: {}
      output:
        ret3250.1: {}
      attr:
        algorithm: mul
    aten::mul_953:
      type: Mul
      input:
        '1115': {}
        ret3250.1: {}
      output:
        ret3251.1: {}
      attr:
        algorithm: mul
    aten::size_3428:
      type: Shape
      input:
        ret3251.1: {}
      output:
        '10109': {}
      attr:
        start: 0
        end: 0
    aten::size_3340:
      type: Shape
      input:
        ret3251.1: {}
      output:
        '10111': {}
      attr:
        start: 1
        end: 1
    aten::mul_954:
      type: Mul
      input:
        ret3251.1: {}
        '1116': {}
      output:
        ret3254.1: {}
      attr:
        algorithm: mul
    aten::linear_3709:
      type: InnerProduct
      input:
        ret3254.1: {}
        '1117': {}
        aten::linear_3709_bias: {}
      output:
        ret3257.1: {}
    aten::view_5746:
      type: View
      input:
        ret3257.1: {}
        '10109': {}
        '10111': {}
      output:
        ret3258.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3046:
      type: Reorder
      input:
        ret3258.1: {}
      output:
        ret3259.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_956:
      type: Mul
      input:
        ret3251.1: {}
        '1118': {}
      output:
        ret3260.1: {}
      attr:
        algorithm: mul
    aten::linear_3710:
      type: InnerProduct
      input:
        ret3260.1: {}
        '1119': {}
        aten::linear_3710_bias: {}
      output:
        ret3263.1: {}
    aten::view_5747:
      type: View
      input:
        ret3263.1: {}
        '10109': {}
        '10111': {}
      output:
        ret3264.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3047:
      type: Reorder
      input:
        ret3264.1: {}
      output:
        ret3265.1: {}
      attr:
        transpose_dims: 1,2
    aten::mul_958:
      type: Mul
      input:
        ret3251.1: {}
        '1120': {}
      output:
        ret3266.1: {}
      attr:
        algorithm: mul
    aten::linear_3711:
      type: InnerProduct
      input:
        ret3266.1: {}
        '1121': {}
        aten::linear_3711_bias: {}
      output:
        ret3269.1: {}
    aten::view_5748:
      type: View
      input:
        ret3269.1: {}
        '10109': {}
        '10111': {}
      output:
        ret3270.1: {}
      attr:
        shape: -1,-1,40,128
    aten::transpose_3048:
      type: Reorder
      input:
        ret3270.1: {}
      output:
        ret3271.1: {}
      attr:
        transpose_dims: 1,2
    aten::size_3049:
      type: Shape
      input:
        ret3265.1: {}
      output:
        '10157': {}
      attr:
        start: 2
        end: 2
    aten::size_3050:
      type: Shape
      input:
        x77.1: {}
      output:
        '10160': {}
      attr:
        start: 2
        end: 2
    aten::add_3386:
      type: Add
      input:
        '10157': {}
        '10160': {}
      output:
        seq_len38.1: {}
    aten::slice_135:
      type: Slice
      input:
        '493': {}
        seq_len38.1: {}
      output:
        '10167': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::slice_175:
      type: Slice
      input:
        '494': {}
        seq_len38.1: {}
      output:
        '10169': {}
      attr:
        axes: 2
        starts: 0
        ends: null
        steps: 1
    aten::size_3051:
      type: Shape
      input:
        ret3259.1: {}
      output:
        '10171': {}
      attr:
        start: 2
        end: 2
    aten::add_3341:
      type: Add
      input:
        '10171': {}
        '10160': {}
      output:
        '10176': {}
    aten::slice_3052:
      type: Slice
      input:
        '10167': {}
        '10160': {}
        '10176': {}
      output:
        '10178': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2396:
      type: Slice
      input:
        '10178': {}
      output:
        '10179': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::slice_3053:
      type: Slice
      input:
        '10169': {}
        '10160': {}
        '10176': {}
      output:
        '10180': {}
      attr:
        axes: 2
        starts: null
        ends: null
        steps: 1
    aten::slice_2397:
      type: Slice
      input:
        '10180': {}
      output:
        '10181': {}
      attr:
        axes: 3
        starts: 0
        ends: 9223372036854775807
        steps: 1
    aten::mul_5749:
      type: Mul
      input:
        ret3259.1: {}
        '10179': {}
      output:
        ret3275.1: {}
      attr:
        algorithm: mul
    aten::size_2398:
      type: Shape
      input:
        ret3259.1: {}
      output:
        '10185': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_254:
      type: Div
      input:
        '10185': {}
        '495': {}
      output:
        ret3277.1: {}
      attr:
        algorithm: div
    aten::slice_2399:
      type: Slice
      input:
        ret3259.1: {}
        ret3277.1: {}
      output:
        ret3278.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2400:
      type: Slice
      input:
        ret3259.1: {}
        ret3277.1: {}
      output:
        ret3279.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5756:
      type: Neg
      input:
        ret3279.1: {}
        aten::neg_5756_mul_val: {}
      output:
        ret3280.1: {}
      attr:
        algorithm: mul
    aten::cat_2531:
      type: Concat
      input:
        ret3280.1: {}
        ret3278.1: {}
      output:
        ret3281.1: {}
      attr:
        axis: -1
    aten::mul_5753:
      type: Mul
      input:
        ret3281.1: {}
        '10181': {}
      output:
        ret3282.1: {}
      attr:
        algorithm: mul
    aten::add_3342:
      type: Add
      input:
        ret3275.1: {}
        ret3282.1: {}
      output:
        args155.1: {}
    aten::mul_5751:
      type: Mul
      input:
        ret3265.1: {}
        '10179': {}
      output:
        ret3283.1: {}
      attr:
        algorithm: mul
    aten::size_2401:
      type: Shape
      input:
        ret3265.1: {}
      output:
        '10207': {}
      attr:
        start: 3
        end: 3
    aten::floor_divide_255:
      type: Div
      input:
        '10207': {}
        '495': {}
      output:
        ret3285.1: {}
      attr:
        algorithm: div
    aten::slice_2402:
      type: Slice
      input:
        ret3265.1: {}
        ret3285.1: {}
      output:
        ret3286.1: {}
      attr:
        axes: 3
        starts: 0
        ends: null
        steps: 1
    aten::slice_2403:
      type: Slice
      input:
        ret3265.1: {}
        ret3285.1: {}
      output:
        ret3287.1: {}
      attr:
        axes: 3
        starts: null
        ends: 9223372036854775807
        steps: 1
    aten::neg_5758:
      type: Neg
      input:
        ret3287.1: {}
        aten::neg_5758_mul_val: {}
      output:
        ret3288.1: {}
      attr:
        algorithm: mul
    aten::cat_2532:
      type: Concat
      input:
        ret3288.1: {}
        ret3286.1: {}
      output:
        ret3289.1: {}
      attr:
        axis: -1
    aten::mul_5754:
      type: Mul
      input:
        ret3289.1: {}
        '10181': {}
      output:
        ret3290.1: {}
      attr:
        algorithm: mul
    aten::add_3343:
      type: Add
      input:
        ret3283.1: {}
        ret3290.1: {}
      output:
        '10225': {}
    aten::cat_3054:
      type: Concat
      input:
        x77.1: {}
        '10225': {}
      output:
        ret3291.1: {}
      attr:
        axis: 2
    aten::cat_3055:
      type: Concat
      input:
        x78.1: {}
        ret3271.1: {}
      output:
        ret3292.1: {}
      attr:
        axis: 2
    aten::transpose_2404:
      type: Reorder
      input:
        ret3291.1: {}
      output:
        ret3293.1: {}
      attr:
        transpose_dims: 2,3
    aten::matmul_5761:
      type: Matmul
      input:
        args155.1: {}
        ret3293.1: {}
      output:
        ret3296.1: {}
    aten::div_295:
      type: Div
      input:
        ret3296.1: {}
        '496': {}
      output:
        ret3297.1: {}
      attr:
        algorithm: div
    aten::add_3344:
      type: Add
      input:
        ret3297.1: {}
        attention_mask0.1: {}
      output:
        attn_weights78.1: {}
    aten::max_335:
      type: Max
      input:
        attn_weights78.1: {}
        '497': {}
      output:
        input78.1: {}
    aten::softmax_2448:
      type: Softmax
      input:
        input78.1: {}
      output:
        '10247': {}
      attr:
        axis: -1
    aten::matmul_5764:
      type: Matmul
      input:
        '10247': {}
        ret3292.1: {}
      output:
        ret3300.1: {}
    aten::transpose_3056:
      type: Reorder
      input:
        ret3300.1: {}
      output:
        ret3301.1: {}
      attr:
        transpose_dims: 1,2
    aten::reshape_5766:
      type: Reshape
      input:
        ret3301.1: {}
        '10109': {}
        '10111': {}
      output:
        ret3302.1: {}
    aten::mul_960:
      type: Mul
      input:
        ret3302.1: {}
        '1122': {}
      output:
        ret3303.1: {}
      attr:
        algorithm: mul
    aten::linear_3712:
      type: InnerProduct
      input:
        ret3303.1: {}
        '1123': {}
        aten::linear_3712_bias: {}
      output:
        ret3306.1: {}
    aten::add_3345:
      type: Add
      input:
        ret3245.1: {}
        ret3306.1: {}
      output:
        ret3307.1: {}
    aten::pow_3057:
      type: Pow
      input:
        ret3307.1: {}
        aten::pow_3057_other: {}
      output:
        ret3308.1: {}
    aten::mean_1052:
      type: ReduceMean
      input:
        ret3308.1: {}
      output:
        ret3309.1: {}
    aten::add_87:
      type: Add
      input:
        ret3309.1: {}
        '485': {}
      output:
        ret3310.1: {}
    aten::rsqrt_5769:
      type: Rsqrt
      input:
        ret3310.1: {}
      output:
        ret3311.1: {}
    aten::mul_5768:
      type: Mul
      input:
        ret3307.1: {}
        ret3311.1: {}
      output:
        ret3312.1: {}
      attr:
        algorithm: mul
    aten::mul_962:
      type: Mul
      input:
        '1124': {}
        ret3312.1: {}
      output:
        ret3313.1: {}
      attr:
        algorithm: mul
    aten::mul_963:
      type: Mul
      input:
        ret3313.1: {}
        '1125': {}
      output:
        ret3314.1: {}
      attr:
        algorithm: mul
    aten::linear_3713:
      type: InnerProduct
      input:
        ret3314.1: {}
        '1126': {}
        aten::linear_3713_bias: {}
      output:
        ret3317.1: {}
    aten::silu_5771:
      type: Swish
      input:
        ret3317.1: {}
      output:
        ret3318.1: {}
    aten::mul_965:
      type: Mul
      input:
        ret3313.1: {}
        '1127': {}
      output:
        ret3319.1: {}
      attr:
        algorithm: mul
    aten::linear_3714:
      type: InnerProduct
      input:
        ret3319.1: {}
        '1128': {}
        aten::linear_3714_bias: {}
      output:
        ret3322.1: {}
    aten::mul_5772:
      type: Mul
      input:
        ret3318.1: {}
        ret3322.1: {}
      output:
        ret3323.1: {}
      attr:
        algorithm: mul
    aten::mul_967:
      type: Mul
      input:
        ret3323.1: {}
        '1129': {}
      output:
        ret3324.1: {}
      attr:
        algorithm: mul
    aten::linear_3715:
      type: InnerProduct
      input:
        ret3324.1: {}
        '1130': {}
        aten::linear_3715_bias: {}
      output:
        ret3327.1: {}
    aten::add_3346:
      type: Add
      input:
        ret3307.1: {}
        ret3327.1: {}
      output:
        ret3328.1: {}
    aten::pow_3058:
      type: Pow
      input:
        ret3328.1: {}
        aten::pow_3058_other: {}
      output:
        ret3329.1: {}
    aten::mean_1053:
      type: ReduceMean
      input:
        ret3329.1: {}
      output:
        ret3330.1: {}
    aten::add_88:
      type: Add
      input:
        ret3330.1: {}
        '485': {}
      output:
        ret3331.1: {}
    aten::rsqrt_5776:
      type: Rsqrt
      input:
        ret3331.1: {}
      output:
        ret3332.1: {}
    aten::mul_5775:
      type: Mul
      input:
        ret3328.1: {}
        ret3332.1: {}
      output:
        ret3333.1: {}
      attr:
        algorithm: mul
    aten::mul_969:
      type: Mul
      input:
        '1131': {}
        ret3333.1: {}
      output:
        ret3334.1: {}
      attr:
        algorithm: mul
    aten::mul_970:
      type: Mul
      input:
        ret3334.1: {}
        '1132': {}
      output:
        ret3335.1: {}
      attr:
        algorithm: mul
    aten::linear_3716:
      type: InnerProduct
      input:
        ret3335.1: {}
        '1133': {}
        aten::linear_3716_bias: {}
      output:
        ret3338.1: {}
'''
        file = open('conf.yaml', 'w')
        file.write(text)
        file.close()
        llamagraph = Graph()
        llamagraph.graph_init('./conf.yaml')
        oldlen = len(llamagraph.nodes)
        p_fusion = PATTERNS['LlamaEmbeddings']()
        llamagraph = p_fusion(llamagraph)
        newlen = len(llamagraph.nodes)
        self.assertTrue(oldlen != newlen)
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['InnerproductReshapeFusion']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['MatMulWithTranspose']()
        llamagraph = p_fusion(llamagraph)
        oldlen = len(llamagraph.nodes)
        p_fusion = PATTERNS['LlamaMatMulWithTranspose']()
        llamagraph = p_fusion(llamagraph)
        newlen = len(llamagraph.nodes)
        self.assertTrue(oldlen != newlen)
        oldlen = len(llamagraph.nodes)
        p_fusion = PATTERNS['InnerproductWithSwish']()
        llamagraph = p_fusion(llamagraph)
        newlen = len(llamagraph.nodes)
        self.assertTrue(oldlen != newlen)
        p_fusion = PATTERNS['SliceMask']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['ArangewithReciprocal']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['InnerproductwithSlice']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['RoraryPosEmb']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['EinsumwithArange']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['RemoveSlice']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['RemoveLastView']()
        llamagraph = p_fusion(llamagraph)
        p_fusion = PATTERNS['RemoveConstantOP']()
        llamagraph = p_fusion(llamagraph)
        oldlen = len(llamagraph.nodes)
        p_fusion = PATTERNS['LlamaRoraryPosEmb']()
        llamagraph = p_fusion(llamagraph)
        newlen = len(llamagraph.nodes)
        oldlen = len(llamagraph.nodes)
        p_fusion = PATTERNS['LlamaPostprocess']()
        llamagraph = p_fusion(llamagraph)
        newlen = len(llamagraph.nodes)
        self.assertTrue(oldlen != newlen)
        

if __name__ == "__main__":
    unittest.main()
