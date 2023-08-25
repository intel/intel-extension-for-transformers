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

plugins = DotDict({})

def register_plugin(name):
    def decorator(cls):
        enable = True if name == "asr" else False
        plugins[name] = {
            'enable': enable,
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
