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
"""Neural Chat Python logging."""

import logging

def configure_logging(log_file="app.log", log_level=logging.INFO):
    """
    Configure logging for the application.

    Parameters:
    - log_file: str, optional, default: "app.log"
        The name of the log file.
    - log_level: int, optional, default: logging.INFO
        The logging level.

    Returns:
    - logger: logging.Logger
        The configured logger instance with specified handlers and formatters.
    """
    logger = logging.getLogger("my_app")
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file, delay=True)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
