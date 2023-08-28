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

import argparse
import json
import time

from typing import List

import requests

from .base_executor import BaseCommandExecutor
from .server_commands import cli_client_command_register
from ..cli.log import logger


__all__ = [
    'TextChatClientExecutor', 'VoiceChatClientExecutor', 'FinetuningClientExecutor'
]


# @cli_client_register(name='neuralchat_client.textchat', description='visit text chat service')
class TextChatClientExecutor(BaseCommandExecutor):
    def __init__(self):
        super(TextChatClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='neuralstudio_client.textchat', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8000, help='server port')
        self.parser.add_argument(
            '--query',
            type=str,
            default=None,
            help='the initial input or context provided to the text generation model',
            required=True)
        self.parser.add_argument(
            '--device', type=str, default='cpu', help='the device type for text generation')
        self.parser.add_argument(
            '--temperature',
            type=float,
            default=0.1,
            help='control the randomness of the generated text, the value should be set between 0 and 1.0')
        self.parser.add_argument(
            '--top_p',
            type=float,
            default=0.75,
            help='the cumulative probability threshold for using in the top-p sampling strategy, \
                  the value should be set between 0 and 1.0')
        self.parser.add_argument(
            '--top_k',
            type=int,
            default=1,
            help='the number of highest probability tokens to consider in the top-k sampling strategy, \
                  the value should be set between 0 and 200')
        self.parser.add_argument(
            '--repetition_penalty',
            type=float,
            default=1.1,
            help='The penalty applied to repeated tokens, the value should be set between 1.0 and 2.0')
        self.parser.add_argument(
            '--max_new_tokens',
            type=int,
            default=128,
            help='The maximum number of new tokens to generate, the value should be set between 32 and 2048')


    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        prompt = args.query
        server_ip = args.server_ip
        port = args.port
        device = args.device
        temperature = args.temperature
        top_p = args.top_p
        top_k = args.top_k
        repetition_penalty = args.repetition_penalty
        max_new_tokens = args.max_new_tokens

        try:
            time_start = time.time()
            res = self(
                prompt=prompt,
                server_ip=server_ip,
                port=port,
                device=device,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens)
            time_end = time.time()
            time_consume = time_end - time_start
            response_dict = res.json()
            print("======= Textchat Client Response =======")
            print(response_dict['response'])
            logger.info("Response time: %f s." % (time_consume))
            return True
        except Exception as e:
            logger.error("Failed to generate text response.")
            logger.error(e)
            return False

    def __call__(self,
                 prompt: str,
                 server_ip: str="127.0.0.1",
                 port: int=8000,
                 device: str='cpu',
                 temperature: float=0.1,
                 top_p: float=0.75,
                 top_k: int=1,
                 repetition_penalty: float=1.1,
                 max_new_tokens: int=128):
        """
        Python API to call an executor.
        """

        url = 'http://' + server_ip + ":" + str(port) + '/v1/chat/completions'
        request = {
            "prompt": prompt,
            "device": device,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens
        }

        res = requests.post(url, json.dumps(request))
        return res


class VoiceChatClientExecutor(BaseCommandExecutor):
    def __init__(self):
        super(VoiceChatClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='neuralstudio_client.voicechat', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8000, help='server port')
        self.parser.add_argument(
            '--audio_input_path', type=str, default=None, help='Input aduio path.')
        self.parser.add_argument(
            '--audio_output_path', type=str, default=None, help='Output aduio path.')
        
    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        server_ip = args.server_ip
        port = args.port
        audio_input_path = args.audio_input_path
        audio_output_path = args.audio_output_path

        try:
            time_start = time.time()
            res = self(
                server_ip=server_ip,
                port=port,
                audio_input_path=audio_input_path,
                audio_output_path=audio_output_path)
            time_end = time.time()
            time_consume = time_end - time_start
            print("======= Voicechat Client Response =======")
            print(res.text)
            logger.info("Response time: %f s." % (time_consume))
            return True
        except Exception as e:
            logger.error("Failed to generate text response.")
            logger.error(e)
            return False
        
    def __call__(self,
                 server_ip: str="127.0.0.1",
                 port: int=8000,
                 audio_input_path: str=None,
                 audio_output_path: str=None):
        url = 'http://' + server_ip + ":" + str(port) + '/v1/voicechat/completions'
        outpath = audio_output_path if audio_output_path is not None else " "
        with open(audio_input_path, "rb") as wav_file:
            files = {
                "file": ("audio.wav", wav_file, "audio/wav"),
                "voice": (None, "pat"),
                "audio_output_path": (None, outpath)
            }
            res = requests.post(url, files=files)
            return res


class FinetuningClientExecutor(BaseCommandExecutor):
    def __init__(self):
        super(FinetuningClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='neuralstudio_client.finetune', add_help=True)
        self.parser.add_argument(
            '--server_ip', type=str, default='127.0.0.1', help='server ip')
        self.parser.add_argument(
            '--port', type=int, default=8000, help='server port')
        self.parser.add_argument(
            '--model_name_or_path', type=str, default=None, help='Model name or model path.')
        self.parser.add_argument(
            '--train_file', type=str, default=None, help='Train dataset file for finetuning.')
        
    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        server_ip = args.server_ip
        port = args.port
        model_name_or_path = args.model_name_or_path
        train_file = args.train_file

        try:
            time_start = time.time()
            res = self(
                server_ip=server_ip,
                port=port,
                model_name_or_path=model_name_or_path,
                train_file=train_file)
            time_end = time.time()
            time_consume = time_end - time_start
            print("======= Finetuning Client Response =======")
            print(res.text)
            logger.info("Response time: %f s." % (time_consume))
            return True
        except Exception as e:
            logger.error("Failed to finetune.")
            logger.error(e)
            return False
        
    def __call__(self,
                 server_ip: str="127.0.0.1",
                 port: int=8000,
                 model_name_or_path: str="facebook/opt-125m",
                 train_file: str=None):
        url = 'http://' + server_ip + ":" + str(port) + '/v1/finetune'
        request = {
            "model_name_or_path": model_name_or_path,
            "train_file": train_file
        }
        res = requests.post(url, json.dumps(request))
        return res



specific_commands = {
    'textchat': ['neuralchat_client text chat command', 'TextChatClientExecutor'],
    'voicechat': ['neuralchat_client voice chat command', 'VoiceChatClientExecutor'],
    'finetune': ['neuralchat_client finetuning command', 'FinetuningClientExecutor'],
}

for com, info in specific_commands.items():
    cli_client_command_register(
        name='neuralchat_client.{}'.format(com),
        description=info[0],
        cls='neural_chat.server.neuralchat_client.{}'.format(info[1]))