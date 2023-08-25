#!/usr/bin/env python
import os
import unittest
from neural_chat.cli.log import logger
import subprocess

class UnitTest(unittest.TestCase):

    def test_text_chat(self):
        logger.info(f'Testing CLI request === Text Chat ===')
        command = 'neuralchat textchat \
                    --query "Tell me about Intel." \
                    --model_name_or_path "./Llama-2-7b-chat-hf"'
        try:
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        print(result.stdout)
        self.assertEqual(result.stdout, 0, msg="Textchat command line test failed.")

    def test_help(self):
        logger.info(f'Testing CLI request === Help ===')
        command = 'neuralchat help'
        try:
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertEqual(result.stdout, 0, msg="Textchat command line test failed.")

    def test_voice_chat(self):
        logger.info(f'Testing CLI request === Voice Chat ===')
        command = 'neuralchat voicechat \
                    --query "Tell me about Intel Xeon Scalable Processors." \
                    --audio_output_path "./response.wav" \
                    --model_name_or_path "./Llama-2-7b-chat-hf"'
        try:
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print("Error while executing command:", e)
        self.assertEqual(result.stdout, 0, msg="Textchat command line test failed.")


if __name__ == "__main__":
    unittest.main()