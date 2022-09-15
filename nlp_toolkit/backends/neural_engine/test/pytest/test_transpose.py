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

import unittest
from nlp_toolkit.backends.neural_engine.compile.graph import Graph
from nlp_toolkit.backends.neural_engine.compile import compile
import numpy as np
import os


class TestTranspose(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_transpose(self):
        model_dir = '/home/tensorflow/inc_ut/engine/bert_mini_int8_original_IR'
        if not os.path.exists(model_dir):
            print(
                "The model dir is not not found, therefore test may not all round"
            )
            return
        model = compile(model_dir)
        model.transpose_mode_int8()

if __name__ == "__main__":
    unittest.main()
