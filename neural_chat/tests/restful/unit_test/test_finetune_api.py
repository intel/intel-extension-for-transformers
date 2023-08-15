#!/usr/bin/env python
import requests
import unittest
from neural_chat.tests.restful.config import HOST, API_FINETUNE
from neural_chat.cli.log import logger


class UnitTest(unittest.TestCase):

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_completions(self):
        logger.info(f'Testing POST request: {self.host+API_FINETUNE}')
        response = requests.post(self.host+API_FINETUNE)
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response text: {}'.format(response.text))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")


if __name__ == "__main__":
    unittest.main()