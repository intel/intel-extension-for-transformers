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
from tensorflow.core.framework import node_def_pb2 
import intel_extension_for_transformers.backends.neural_engine.compile.tf_utils as util 


class TestTfUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_create_tf_node(self):
        test_node = util.create_tf_node('Reshape', 'test_name', ['input_0'])
        self.assertEqual('Reshape', test_node.op)
        self.assertEqual('test_name', test_node.name)
        self.assertSequenceEqual(['input_0'], test_node.input)


if __name__ == "__main__":
    unittest.main()
