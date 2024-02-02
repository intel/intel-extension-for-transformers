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

import os
from dotenv import load_dotenv

from functools import lru_cache

from pydantic import BaseSettings

from intel_extension_for_transformers.utils import logger

load_dotenv()


class Settings(BaseSettings):
    mysql_user: str = os.environ.get("MYSQL_USER", "root")
    mysql_password: str = os.environ.get("MYSQL_PASSWORD")
    mysql_host: str = os.environ.get("MYSQL_HOST", "localhost")
    mysql_db: str = os.getenv("MYSQL_DB", "inc")
    google_oauth_client_id: str = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
    google_oauth_client_secret: str = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
    github_oauth_client_id: str = os.getenv("GITHUB_OAUTH_CLIENT_ID", "")
    github_oauth_client_secret: str = os.getenv("GITHUB_OAUTH_CLIENT_SECRET", "")
    facebook_oauth_client_id: str = os.getenv("FACEBOOK_OAUTH_CLIENT_ID", "")
    facebook_oauth_client_secret: str = os.getenv("FACEBOOK_OAUTH_CLIENT_SECRET", "")
    microsoft_oauth_client_id: str = os.getenv("MICROSOFT_OAUTH_CLIENT_ID", "")
    microsoft_oauth_client_secret: str = os.getenv("MICROSOFT_OAUTH_CLIENT_SECRET", "")


@lru_cache()
def get_settings() -> BaseSettings:
    logger.info("Loading config settings from the environment...")
    return Settings()
