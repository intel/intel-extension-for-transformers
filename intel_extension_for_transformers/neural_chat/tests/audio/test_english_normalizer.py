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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.audio.utils.english_normalizer import EnglishNormalizer
import unittest

class TestTTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.normalizer = EnglishNormalizer()

    @classmethod
    def tearDownClass(self):
        pass

    def test_correct_number(self):
        text = "3000 people among 1.2 billion people"
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "three thousand people among one point two billion people")

    def test_correct_abbreviation(self):
        text = "SATG AIA a great department"
        result = self.normalizer.correct_abbreviation(text)
        self.assertEqual(result, "ess Eigh tee jee Eigh I Eigh a great department")

if __name__ == "__main__":
    unittest.main()
