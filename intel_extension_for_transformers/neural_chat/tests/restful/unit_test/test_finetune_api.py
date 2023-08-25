#!/usr/bin/env python
import requests
import unittest
import shutil
import os
import json
from intel_extension_for_transformers.neural_chat.tests.restful.config import HOST, API_FINETUNE
from intel_extension_for_transformers.neural_chat.cli.log import logger


json_data = \
"""
[
    {"instruction": "Generate a slogan for a software company", "input": "", "output": "The Future of Software is Here"},
    {"instruction": "Provide the word that comes immediately after the.", "input": "He threw the ball over the fence.", "output": "fence."}
]
"""
test_data_file = '/test.json'


class UnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        with open(test_data_file, mode='w') as f:
            f.write(json_data)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./tmp', ignore_errors=True)
        os.remove(test_data_file)

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_finetune(self):
        logger.info(f'Testing POST request: {self.host+API_FINETUNE}')
        request = {
            "model_name_or_path": "facebook/opt-125m",
            "train_file": "/test.json"
        }
        response = requests.post(self.host+API_FINETUNE, json.dumps(request))
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response text: {}'.format(response.text))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")


if __name__ == "__main__":
    unittest.main()