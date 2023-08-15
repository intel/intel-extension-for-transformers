#!/usr/bin/env python
import requests
import unittest
from neural_chat.tests.restful.config import HOST, API_TEXT2IMAGE
from neural_chat.server.restful.response import ImageResponse
from neural_chat.cli.log import logger


class UnitTest(unittest.TestCase):

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_completions(self):
        logger.info(f'Testing POST request: {self.host+API_TEXT2IMAGE}')
        text = "A running horse."
        response = requests.post(self.host+API_TEXT2IMAGE, data=text)
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response text: {}'.format(response.response))
        logger.info('Response image: {}'.format(response.image))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")


if __name__ == "__main__":
    unittest.main()