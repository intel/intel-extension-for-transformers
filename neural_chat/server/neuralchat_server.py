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
import sys
import os
from typing import List


import uvicorn
import yaml
import logging
from yacs.config import CfgNode
from fastapi import FastAPI
from fastapi import APIRouter
from starlette.middleware.cors import CORSMiddleware

from .base_executor import BaseCommandExecutor
from .server_commands import cli_server_register

from neural_chat.cli.log import logger
from .restful.api import setup_router
from neural_chat.config import PipelineConfig
from neural_chat.chatbot import build_chatbot


__all__ = ['NeuralChatServerExecutor']

app = FastAPI(
    title="NeuralChat Serving API", description="Api", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

api_router = APIRouter()




def get_config(config_file: str):
    """Get config from yaml config file.

    Args:
        config_file (str): config_file

    Returns:
        CfgNode:
    """
    with open(config_file, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    return config

@cli_server_register(name='neuralchat_server.start', description='Start the service')
class NeuralChatServerExecutor(BaseCommandExecutor):
    def __init__(self):
        super(NeuralChatServerExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(
            prog='neuralchat_server.start', add_help=True)
        self.parser.add_argument(
            "--config_file",
            action="store",
            help="yaml file of the server",
            default=None,
            required=True)

        self.parser.add_argument(
            "--log_file",
            action="store",
            help="log file",
            default="./log/neuralchat.log")

    def init(self, config):
        """System initialization.

        Args:
            config (CfgNode): config object

        Returns:
            bool:
        """
        plugin_list = list(plugin for plugin in config.plugins_list)
        params = {}
        # Model configuration
        if config.model_name:
            params["model_name_or_path"] = config.model_name
        # Audio plugin configuration
        if "audio" in plugin_list:
            params["audio_input"] = config.audio.audio_input
            params["audio_output"] = config.audio.audio_output
        # Retrieval plugin configuration
        if "retrieval" in plugin_list:
            params["retrieval_type"] = config.retrieval.retrieval_type
            script_dir = os.path.dirname(os.path.abspath(__file__))
            retrieval_document_path = os.path.join(script_dir, config.retrieval.retrieval_document_path)
            params["retrieval_document_path"] = retrieval_document_path
        # Caching plugin configuration
        if "caching" in plugin_list:
            params["cache_chat_config_file"] = config.caching.cache_chat_config_file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            retrieval_document_path = os.path.join(script_dir, config.caching.cache_embedding_model_dir)
            params["cache_embedding_model_dir"] = retrieval_document_path
        # Other plugins configurations
        for plugin in ["memory_controller", "intent_detection", "safety_checker"]:
            if plugin in config.plugins_list:
                params[plugin] = True
        pipeline_config = PipelineConfig(**params)
        self.chatbot = build_chatbot(pipeline_config)

        # init api
        api_list = list(task for task in config.tasks_list)
        api_router = setup_router(api_list, self.chatbot)
        app.include_router(api_router)
        return True


    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        try:
            self(args.config_file, args.log_file)
        except Exception as e:
            logger.error("Failed to start server.")
            logger.error(e)
            sys.exit(-1)

    def __call__(self,
                 config_file: str="./conf/neuralchat.yaml",
                 log_file: str="./log/neuralchat.log"):
        """
        Python API to call an executor.
        """
        config = get_config(config_file)
        if self.init(config):
            logging.basicConfig(filename=log_file, level=logging.INFO)
            uvicorn.run(app, host=config.host, port=config.port)
