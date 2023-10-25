# !/usr/bin/env python
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

import re
import os
import datetime
from datetime import timezone, timedelta


def enforce_stop_tokens(text: str) -> str:
    """Cut off the text as soon as any stop words occur."""
    stopwords = ["</s"]
    return re.split("|".join(stopwords), text)[0]


def get_current_time() -> str:
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    cur_time = utc_now.astimezone(SHA_TZ).strftime("%Y/%m/%d")
    return cur_time
