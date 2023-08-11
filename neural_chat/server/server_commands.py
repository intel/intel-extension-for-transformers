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

from typing import List
import sys
from ..utils.command import NeuralChatCommandDict


__all__ = [
    'NeuralChatServerBaseCommand',
    'NeuralChatServerHelpCommand',
    'NeuralChatClientBaseCommand',
    'NeuralChatClientHelpCommand',
    'neuralchat_server_commands',
    'neuralchat_client_commands'
]


neuralchat_server_commands = NeuralChatCommandDict()
neuralchat_client_commands = NeuralChatCommandDict()

def cli_server_register(name: str, description: str=''):
    def _wrapper(command):
        items = name.split('.')

        com = neuralchat_server_commands
        for item in items:
            com = com[item]
        com['command'] = command
        if description:
            com['description'] = description
        return command

    return _wrapper


def get_server_command(name: str):
    items = name.split('.')
    com = neuralchat_server_commands
    for item in items:
        com = com[item]

    return com['command']


def cli_client_register(name: str, description: str=''):
    def _wrapper(command):
        items = name.split('.')

        com = neuralchat_client_commands
        for item in items:
            com = com[item]
        com['command'] = command
        if description:
            com['description'] = description
        return command

    return _wrapper


def get_client_command(name: str):
    items = name.split('.')
    com = neuralchat_client_commands
    for item in items:
        com = com[item]

    return com['command']


def _neuralchat_execute(commands, command_name_prefix):
    idx = 0
    for _argv in ([command_name_prefix] + sys.argv[1:]):
        if _argv not in commands:
            break
        idx += 1
        commands = commands[_argv]

    try:
        status = 0 if commands['command']().execute(sys.argv[idx:]) else 1
    except Exception as e:
        print("An error occurred on neuralchat command execution:", str(e))
        status = 1

    return status

def neuralchat_server_execute():
    return _neuralchat_execute(neuralchat_server_commands, 'neuralchat_server')

def neuralchat_client_execute():
    return _neuralchat_execute(neuralchat_client_commands, 'neuralchat_client')


@cli_server_register(name='neuralchat_server')
class NeuralChatServerBaseCommand:
    def execute(self, argv: List[str]):
        help = get_server_command('neuralchat_server.help')
        return help().execute(argv)


@cli_server_register(name='neuralchat_server.help', description='Show help for commands.')
class NeuralChatServerHelpCommand:
    def execute(self, argv: List[str]) -> bool:
        msg = 'Usage:\n'
        msg += '    neuralchat_server <command> <options>\n\n'
        msg += 'Commands:\n'
        for command, detail in neuralchat_server_commands['neuralchat_server'].items():
            if command.startswith('_'):
                continue

            if 'description' not in detail:
                continue
            msg += '    {:<15}        {}\n'.format(command, detail['description'])
        print(msg)
        return True


@cli_client_register(name='neuralchat_client')
class NeuralChatClientBaseCommand:
    def execute(self, argv: List[str]):
        help = get_client_command('neuralchat_client.help')
        return help().execute(argv)


@cli_client_register(name='neuralchat_client.help', description='Show help for commands.')
class NeuralChatClientHelpCommand:
    def execute(self, argv: List[str]):
        msg = 'Usage:\n'
        msg += '    neuralchat_client <command> <options>\n\n'
        msg += 'Commands:\n'
        for command, detail in neuralchat_client_commands['neuralchat_client'].items():
            if command.startswith('_'):
                continue

            if 'description' not in detail:
                continue
            msg += '    {:<15}        {}\n'.format(command, detail['description'])

        print(msg)
        return True
