#!/usr/bin/env python
import requests
import unittest
from datasets import Dataset, Audio
from neural_chat.tests.restful.config import HOST, API_AUDIO
from neural_chat.cli.log import logger


class UnitTest(unittest.TestCase):

    def __init__(self, *args):
        super(UnitTest, self).__init__(*args)
        self.host = HOST

    def test_voicechat_text_out(self):
        logger.info(f'Testing POST request: {self.host+API_AUDIO} with text output.')
        audio_path = "../../../assets/audio/pat.wav"

        with open(audio_path, "rb") as wav_file:
            files = {
                "file": ("audio.wav", wav_file, "audio/wav"),
                "voice": (None, "pat"),
                "audio_output_path": (None, " ")
            }
            response = requests.post(self.host+API_AUDIO, files=files, verify=False)

            logger.info('Response status code: {}'.format(response.status_code))
            logger.info('Response text: {}'.format(response.text))
            self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")

    def test_voicechat_audio_out(self):
        logger.info(f'Testing POST request: {self.host+API_AUDIO} with audio output.')
        audio_path = "../../../assets/audio/pat.wav"

        with open(audio_path, "rb") as wav_file:
            files = {
                "file": ("audio.wav", wav_file, "audio/wav"),
                "voice": (None, "pat"),
                "audio_output_path": (None, "./response.wav")
            }
            response = requests.post(self.host+API_AUDIO, files=files, verify=False)

            logger.info('Response status code: {}'.format(response.status_code))
            logger.info('Response text: {}'.format(response.text))
            self.assertEqual(response.status_code, 200, msg="Abnormal response status code.")
            

if __name__ == "__main__":
    unittest.main()
