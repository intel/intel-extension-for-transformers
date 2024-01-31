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


import unittest
from intel_extension_for_transformers.neural_chat.utils.dotdict import (
    deep_get, deep_set, DotDict
)


class TestServerCommand(unittest.TestCase):

    def test_deep_get(self):
        person = {'person':{'name':{'first':'John'}}}
        res = deep_get(person, "person.name.first")
        self.assertEqual(res, 'John')

    def test_deep_set(self):
        person = {'person':{'name':{'first':'John'}}}
        deep_set(person, "person.sex", 'male')
        self.assertEqual(person, {'person': {'name': {'first': 'John'}, 'sex': 'male'}})

    def test_DotDict(self):
        test_dotdict = DotDict()
        self.assertEqual(test_dotdict, {})

        test_dotdict2 = DotDict( {'person':{'name':{'first':'John'}}})
        test_dotdict2.__setitem__(key='person2', value=[{'sex': 'male'}])
        self.assertEqual(
            test_dotdict2,
            {'person': {'name': {'first': 'John'}}, 'person2': {'sex': 'male'}})

        test_dotdict2.__setitem__(key='person3', value=[{'sex': 'male'},{'age': 20}])
        self.assertEqual(
            test_dotdict2,
            {
                'person': {'name': {'first': 'John'}},
                'person2': {'sex': 'male'},
                'person3': {'sex': 'male', 'age': 20}})

        test_dotdict2.__setstate__({'test_state': 'active'})
        res = test_dotdict2.__getstate__()
        self.assertEqual(res, {'test_state': 'active'})



if __name__ == "__main__":
    unittest.main()
