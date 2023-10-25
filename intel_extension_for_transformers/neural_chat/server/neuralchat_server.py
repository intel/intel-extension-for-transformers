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

from ..cli.log import logger
from .restful.api import setup_router
from ..config import PipelineConfig, LoadingModelConfig
from ..chatbot import build_chatbot
from ..plugins import plugins

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
        device = config.get("device", "auto")
        model_name_or_path = config.get("model_name_or_path", "meta-llama/Llama-2-7b-hf")
        ipex_int8 = config.get("ipex_int8", False)
        tokenizer_name_or_path = config.get("tokenizer_name_or_path", model_name_or_path)

        # Update plugins based on YAML configuration
        for plugin_name, plugin_config in plugins.items():
            yaml_config = config.get(plugin_name, {})
            if yaml_config.get("enable"):
                plugin_config["enable"] = True
                plugin_config["args"] = yaml_config.get("args", {})
 
        loading_config = LoadingModelConfig(ipex_int8=ipex_int8)
        # Create a dictionary of parameters for PipelineConfig
        params = {
            "model_name_or_path": model_name_or_path,
            "device": device,
            "plugins": plugins,
            "loading_config": loading_config,
            "tokenizer_name_or_path": tokenizer_name_or_path
        }

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
            try:
                uvicorn.run(app, host=config.host, port=config.port)
            except Exception as e:
                print(f"Error starting uvicorn: {str(e)}")
