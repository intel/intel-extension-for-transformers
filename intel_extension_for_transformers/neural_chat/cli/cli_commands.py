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

import sys, os
import argparse
from typing import List
from ..utils.command import NeuralChatCommandDict
from ..utils.common import is_audio_file
from .base_executor import BaseCommandExecutor
from ..config import PipelineConfig, TextGenerationFinetuningConfig
from ..config import ModelArguments, DataArguments, FinetuningArguments
from ..plugins import plugins
from transformers import TrainingArguments
from ..chatbot import build_chatbot, finetune_model
from ..config_logging import configure_logging
logger = configure_logging()

__all__ = ['BaseCommand', 'HelpCommand', 'TextVoiceChatExecutor', 'FinetuingExecutor']

neuralchat_commands = NeuralChatCommandDict()

def get_command(name: str):
    items = name.split('.')
    com = neuralchat_commands
    for item in items:
        com = com[item]

    return com['_command']

def cli_register(name: str, description: str=''):
    def _warpper(command):
        items = name.split('.')

        com = neuralchat_commands
        for item in items:
            com = com[item]
        com['_command'] = command
        if description:
            com['description'] = description
        return command

    return _warpper

def command_register(name: str, description: str='', cls: str=''):
    items = name.split('.')
    com = neuralchat_commands
    for item in items:
        com = com[item]
    com['_command'] = cls
    if description:
        com['description'] = description

def neuralchat_execute():
    com = neuralchat_commands

    idx = 0
    for _argv in (['neuralchat'] + sys.argv[1:]):
        if _argv not in com:
            break
        idx += 1
        com = com[_argv]

    if not callable(com['_command']):
        i = com['_command'].rindex('.')
        module, cls = com['_command'][:i], com['_command'][i + 1:]
        exec("from {} import {}".format(module, cls))
        com['_command'] = locals()[cls]
    status = 0 if com['_command']().execute(sys.argv[idx:]) else 1
    return status

@cli_register(name='neuralchat')
class BaseCommand:
    """
    BaseCommand class serving as a foundation for other neuralchat commands.

    This class provides a common structure for neuralchat commands. It includes a
    default implementation of the execute method, which acts as a fallback and
    invokes the 'neuralchat.help' command to provide assistance to users when
    no specific command is provided.

    Attributes:
        None

    Methods:
        execute(argv): Executes the fallback 'neuralchat.help' command and
                       returns its execution result.

    Usage example:
        base_command = BaseCommand()
        base_command.execute([])
    """
    def execute(self, argv: List[str]) -> bool:
        help = get_command('neuralchat.help')
        return help().execute(argv)


@cli_register(name='neuralchat.help', description='Show help for neuralchat commands.')
class HelpCommand:
    """
    HelpCommand class for displaying help about available neuralchat commands.

    This class provides the functionality to display a list of available neuralchat
    commands and their descriptions. It helps users understand how to use different
    commands provided by the neuralchat package.

    Attributes:
        None

    Methods:
        execute(argv): Executes the help display and returns a success status.
    """
    def execute(self, argv: List[str]) -> bool:
        msg = 'Usage:\n'
        msg += '    neuralchat <command> <options>\n\n'
        msg += 'Commands:\n'
        for command, detail in neuralchat_commands['neuralchat'].items():
            if command.startswith('_'):
                continue

            if 'description' not in detail:
                continue
            msg += '    {:<15}        {}\n'.format(command,
                                                   detail['description'])

        print(msg)
        return True


class TextVoiceChatExecutor(BaseCommandExecutor):
    """
    TextVoiceChatExecutor class for executing text-based or voice-based conversations with a chatbot.

    This class extends the BaseCommandExecutor class and provides functionality for
    interacting with a chatbot through the command line or the Python API. It initializes
    the necessary components, including the argument parser and the chatbot instance.

    Attributes:
        parser (argparse.ArgumentParser): An argument parser for command-line input.
        config (PipelineConfig): Configuration instance for the chatbot.

    Methods:
        execute(argv): Execute the chatbot using command-line arguments.
        __call__(prompt): Python API for calling the chatbot executor.

    """
    def __init__(self):
        """
        Initializes the TextVoiceChatExecutor class.

        This constructor sets up the necessary components for the chatbot executor.
        It creates a command-line argument parser, initializes the configuration,
        initializes the chatbot instance, and builds the chatbot model.
        """
        super().__init__()
        self.parser = argparse.ArgumentParser(
            prog='neuralchat.predict', add_help=True)
        self.parser.add_argument(
            '--query', type=str, default=None, help='Prompt text or audio file.')
        self.parser.add_argument(
            '--model_name_or_path', type=str, default=None, help='Model name or path.')
        self.parser.add_argument(
            '--output_audio_path', type=str, default=None, help='Audio output path if the prompt is audio file.')
        self.parser.add_argument(
            '--device', type=str, default=None, help='Specify chat on which device.')

    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry point.
        """
        parser_args = self.parser.parse_args(argv)

        prompt = parser_args.query
        model_name = parser_args.model_name_or_path
        output_audio_path = parser_args.output_audio_path
        device = parser_args.device
        if os.path.exists(prompt):
            if is_audio_file(prompt):
                plugins.asr.enable = True
                plugins.tts.enable = True
                if output_audio_path:
                    plugins.tts.args["output_audio_path"]=output_audio_path

        if model_name:
            self.config = PipelineConfig(model_name_or_path=model_name, plugins=plugins, device=device)
        else:
            self.config = PipelineConfig(plugins=plugins)
        self.chatbot = build_chatbot(self.config)
        try:
            res = self(prompt)
            logger.info(res)
            return True
        except Exception as e:  # pragma: no cover
            logger.info("TextVoiceChatExecutor Exception: {}".format(e))
            return False

    def __call__(
            self,
            prompt: str):
        """
            Python API to call an executor.
        """
        result = self.chatbot.chat(prompt)
        return result

class FinetuingExecutor(BaseCommandExecutor):
    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser(
            prog='neuralchat.finetune', add_help=True)
        self.parser.add_argument(
            '--base_model', type=str, default=None, help='Base model path or name for finetuning.')
        self.parser.add_argument(
            '--device', type=str, default=None, help='Specify finetune model on which device.')
        self.parser.add_argument(
            '--train_file', type=str, default=None, help='Specify train file path.')
        self.parser.add_argument(
            '--max_steps', type=str, default=None, help='Specify max steps of finetuning.')

    def execute(self, argv: List[str]) -> bool:
        """
            Command line entry.
        """
        parser_args = self.parser.parse_args(argv)
        base_model = parser_args.base_model
        device = parser_args.device
        train_file = parser_args.train_file
        max_steps = parser_args.max_steps

        model_args = ModelArguments(model_name_or_path=base_model)
        data_args = DataArguments(train_file=train_file)
        training_args = TrainingArguments(
            output_dir='./tmp',
            do_train=True,
            max_steps=max_steps,
            overwrite_output_dir=True
        )
        finetune_args = FinetuningArguments(device=device)
        self.finetuneCfg = TextGenerationFinetuningConfig(model_args, data_args, training_args, finetune_args)
        try:
            res = self()
            logger.info(res)
            return True
        except Exception as e:  # pragma: no cover
            logger.info("FinetuingExecutor Exception: {}".format(e))
            return False

    def __call__(self):
        """
            Python API to call an executor.
        """
        finetune_model(self.finetuneCfg)

specific_commands = {
    'predict': ['neuralchat text/voice chat command', 'TextVoiceChatExecutor'],
    'finetune': ['neuralchat finetuning command', 'FinetuingExecutor'],
}

for com, info in specific_commands.items():
    command_register(
        name='neuralchat.{}'.format(com),
        description=info[0],
        cls='intel_extension_for_transformers.neural_chat.cli.cli_commands.{}'.format(info[1]))
