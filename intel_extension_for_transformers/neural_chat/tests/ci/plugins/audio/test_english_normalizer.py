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
        text = "3000 people among 1.2 billion people under -40 degrees."
        result = self.normalizer.correct_abbreviation(text)
        result = self.normalizer.correct_number(result)
        self.assertEqual(result, "three thousand people among one point two billion people under minus forty degrees.")

    def test_correct_abbreviation(self):
        text = "TTS a great technology."
        result = self.normalizer.correct_abbreviation(text)
        self.assertEqual(result, "tee tee ess a great technology.")

    def test_correct_year(self):
        text = "In 1986, there are more than 2000 people participating in that party."
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "In nineteen eightysix, there are more than two thousand people participating in that party.")

    def test_correct_ordinal(self):
        text = "1st 2nd 3rd 4th 5th 11th 12th 21st 22nd"
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "first second third fourth fifth eleventh twelfth twenty first twenty second.")

    def test_correct_conjunctions(self):
        text = "CVPR-15 ICML-21 PM2.5"
        text = self.normalizer.correct_abbreviation(text)
        result = self.normalizer.correct_number(text)
        self.assertEqual(result, "cee vee pea ar fifteen eye cee em el twenty-one pea em two point five.")

if __name__ == "__main__":
    unittest.main()
