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

from .utils.dotdict import DotDict

class GlobalPlugins:
    def __init__(self):
        self.reset_plugins()

    def reset_plugins(self):
        self.plugins = DotDict({
            "tts": {"enable": False, "class": None, "args": {}, "instance": None},
            "tts_chinese": {"enable": False, "class": None, "args": {}, "instance": None},
            "asr": {"enable": False, "class": None, "args": {}, "instance": None},
            "asr_chinese": {"enable": False, "class": None, "args": {}, "instance": None},
            "retrieval": {"enable": False, "class": None, "args": {}, "instance": None},
            "cache": {"enable": False, "class": None, "args": {}, "instance": None},
            "safety_checker": {"enable": False, "class": None, "args": {}, "instance": None},
            "ner": {"enable": False, "class": None, "args": {}, "instance": None},
            "face_animation": {"enable": False, "class": None, "args": {}, "instance": None},
            "image2image": {"enable": False, "class": None, "args": {}, "instance": None},
        })

global_plugins = GlobalPlugins()
plugins = global_plugins.plugins

def register_plugin(name):
    def decorator(cls):
        plugins[name] = {
            'enable': True,
            'class': cls,
            'args': {},
            'instance': None
        }
        return cls
    return decorator

def is_plugin_enabled(plugin_name):
    if plugin_name in plugins and plugins[plugin_name]['enable']:
        return True
    return False

def get_plugin_instance(plugin_name):
    if is_plugin_enabled(plugin_name) and plugins[plugin_name]['instance']:
        return plugins[plugin_name]['instance']
    return None

def get_plugin_arguments(plugin_name):
    if plugin_name in plugins and plugins[plugin_name]['enable']:
        return plugins[plugin_name]['args']
    return None

def get_registered_plugins():
    registered_plugins = []
    for plugin_name, _ in plugins.items():
        registered_plugins.append(plugin_name)
    return registered_plugins

def get_all_plugins():
    return ["tts", "tts_chinese", "asr", "asr_chinese", "retrieval", "cache", "safety_checker", "ner", "ner_int",
            "face_animation"]
