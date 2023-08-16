#!/usr/bin/env python
import requests
import unittest
from neural_chat.tests.restful.config import HOST, API_COMPLETION, API_CHAT_COMPLETION
from neural_chat.server.restful.openai_protocol import ChatCompletionRequest
from neural_chat.cli.log import logger


class UnitTest(unittest.TestCase):

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_completions(self):
        logger.info(f'Testing POST request: {self.host+API_COMPLETION}')
        request = ChatCompletionRequest(
            prompt="This is a test."
        )
        response = requests.post(self.host+API_COMPLETION, data=request)
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response text: {}'.format(response.choices.text))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")

    def test_chat_completions(self):
        logger.info(f'Testing POST request: {self.host+API_CHAT_COMPLETION}')
        request = ChatCompletionRequest(
            prompt="This is a test."
        )
        response = requests.post(self.host+API_CHAT_COMPLETION, data=request)
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response text: {}'.format(response.choices.message.content))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")


if __name__ == "__main__":
    unittest.main()