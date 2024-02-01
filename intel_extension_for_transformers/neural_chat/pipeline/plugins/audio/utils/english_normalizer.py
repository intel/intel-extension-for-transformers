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

from num2words import num2words
import re
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)
class EnglishNormalizer:
    def __init__(self):
        self.correct_dict = {
            "A": "eigh",
            "B": "bee",
            "C": "cee",
            "D": "dee",
            "E": "yee",
            "F": "ef",
            "G": "jee",
            "H": "aitch",
            "I": "eye",
            "J": "jay",
            "K": "kay",
            "L": "el",
            "M": "em",
            "N": "en",
            "O": "o",
            "P": "pea",
            "Q": "cue",
            "R": "ar",
            "S": "ess",
            "T": "tee",
            "U": "u",
            "V": "vee",
            "W": "doubleliu",
            "X": "ex",
            "Y": "wy",
            "Z": "zed",
            ".": "point",
        }

    def correct_abbreviation(self, text):
        # TODO mixed abbreviation or proper noun like i7, ffmpeg, BTW should be supported

        words = re.split(r' |_|/|\*|\#', text)  # ignore the characters that not break sentence
        results = []
        for idx, word in enumerate(words):
            if word.startswith("-"):    # bypass negative number
                parts = [word]
            else:
                parts = word.split('-')
            for w in parts:
                if w.isupper(): # W3C is also upper
                    for c in w:
                        if c in self.correct_dict:
                            results.append(self.correct_dict[c])
                        else:
                            results.append(c)
                else:
                    results.append(w)
        return " ".join(results)

    def correct_number(self, text):
        """Ignore the year or other exception right now"""
        words = text.split()
        results = []
        prepositions_year = ["in", "on"]
        prev = ""
        ordinal_pattern = re.compile("^.*[0-9](st|nd|rd|th)$")
        for idx, word in enumerate(words):
            suffix = ""
            if len(word) > 0 and word[-1] in [",", ".", "?", "!"]:
                suffix = word[-1]
                word = word[:-1]
            if word.isdigit(): # if word is positive integer, it must can be num2words
                try:
                    potential_year = int(word)
                    if prev.lower() in prepositions_year and potential_year < 2999 and potential_year > 1000 \
                          and potential_year % 1000 != 0:
                        word = num2words(word, to="year")
                        word = word.replace("-", "") # nineteen eighty-seven => nineteen eightyseven
                    else:
                        word = num2words(word)
                except Exception as e:
                    logging.info("num2words fail with word: %s and exception: %s", word, e)
            else:
                try:
                    val = int(word)
                    word = num2words(word)
                except ValueError:
                    try:
                        val = float(word)
                        word = num2words(word)
                    except ValueError:
                        # print("not a number, fallback to original word")
                        pass

            if ordinal_pattern.search(word):
                word = num2words(word[:-2], to='ordinal').replace("-", " ")
            word = word + suffix
            results.append(word)
            prev = word
        results = " ".join(results)
        # if the text is not truncated correctly by early stop token, then manually add one.
        if len(results) > 0 and results[-1] not in [",", ".", "?", "!"]:
            results += "."
        return results
