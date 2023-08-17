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
from neural_chat.cli.log import logger


__all__ = [
    'TextChatClientExecutor', 'VoiceChatClientExecutor', 'RetrievalClientExecutor',
    'Text2ImageClientExecutor', 'FinetuningClientExecutor'
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
            logger.info("Text generation duration: %f s." %
                        (response_dict['result']['duration']))
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


specific_commands = {
    'textchat': ['neuralchat_client text chat command', 'TextChatClientExecutor'],
    # 'voicechat': ['neuralchat_client voice chat command', 'VoiceChatExecutor'],
    # 'finetune': ['neuralchat_client finetuning command', 'FinetuingExecutor'],
}

for com, info in specific_commands.items():
    cli_client_command_register(
        name='neuralchat_client.{}'.format(com),
        description=info[0],
        cls='neural_chat.server.neuralchat_client.{}'.format(info[1]))