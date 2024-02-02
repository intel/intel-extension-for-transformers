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

from intel_extension_for_transformers.neural_chat.pipeline.plugins.memory.memory import Memory, Buffer_Memory
import unittest

class TestMemory(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_memory(self):
        query ='hello'
        answer = "Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"
        memory = Memory()
        memory.add(query, answer)
        context = memory.get()
        text = "User Query: hello"
        self.assertIn(text, context)

    def test_buffer_memory(self):
        query ='hello'
        answer = "Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?"
        buffer_memory = Buffer_Memory()
        buffer_memory.add(query, answer)
        context = buffer_memory.get()
        text = "User Query: hello"
        self.assertIn(text, context)

if __name__ == "__main__":
    unittest.main()
