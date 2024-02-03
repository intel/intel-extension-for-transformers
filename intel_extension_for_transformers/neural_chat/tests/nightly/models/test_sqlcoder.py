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

from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
import unittest

class TestSQLCoderModel(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_generate_sql(self):
        config = PipelineConfig(
            model_name_or_path="/tf_dataset2/models/nlp_toolkit/sqlcoder2")
        chatbot = build_chatbot(config=config)
        result = chatbot.predict("Do we get more revenue from customers in New York compared to customers \
            in San Francisco? Give me the total revenue for each city, and the difference between the two.")
        self.assertIn("Given the database schema, here is the SQL query that answers the question", result)
        self.assertIn("SELECT", result)
        self.assertIn("WHERE", result)

if __name__ == "__main__":
    unittest.main()
