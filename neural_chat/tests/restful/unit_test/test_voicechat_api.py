#!/usr/bin/env python
import requests
import unittest
from datasets import Dataset, Audio
from neural_chat.tests.restful.config import HOST, API_ASR, API_TTS
from neural_chat.cli.log import logger


class UnitTest(unittest.TestCase):

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_asr(self):
        logger.info(f'Testing POST request: {self.host+API_ASR}')
        audio_path = "../../../assets/audio/pat.wav"
        audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
        waveform = audio_dataset[0]["audio"]['array']
        response = requests.post(self.host+API_ASR, data=waveform)
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response text: {}'.format(response.text))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")
        self.assertEqual(response.text.lower(), "Welcome to Neural Chat".lower(), msg="Wrong text generated.")

    def test_tts(self):
        logger.info(f'Testing POST request: {self.host+API_TTS}')
        request = "Welcome to Neural Chat"
        response = requests.post(self.host+API_TTS, data=request)
        logger.info('Response status code: {}'.format(response.status_code))
        logger.info('Response: {}'.format(response))
        self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")


if __name__ == "__main__":
    unittest.main()