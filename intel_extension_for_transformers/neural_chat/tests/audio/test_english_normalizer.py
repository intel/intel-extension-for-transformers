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
        text = "3000 people among 1.2 billion people."
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "three thousand people among one point two billion people.")

    def test_correct_abbreviation(self):
        text = "TTS a great technology."
        result = self.normalizer.correct_abbreviation(text)
        self.assertEqual(result, "tee tee ess a great technology.")

    def test_correct_year(self):
        text = "In 1986, there are more than 2000 people participating in that party."
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "In nineteen eighty-six, there are more than two thousand people participating in that party.")

if __name__ == "__main__":
    unittest.main()
