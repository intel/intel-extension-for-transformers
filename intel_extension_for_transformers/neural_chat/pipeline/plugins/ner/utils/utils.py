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


def construct_default_prompt(query: str, cur_time: str) -> str:
    prompt = """Please determine the precise time mentioned in the user's query. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' If the user query does not include any time reference, please reply with 'None'.
    \n\n###Current Time:\n{}\n\nUser Query:\n{}\n\nResponse:\n""".format(cur_time, query)

    return prompt


def construct_refined_prompt(query: str, cur_time: str) -> str:
    prompt = """### Instruction: Please thoughtfully identify the precise time range mentioned in the user's query based on the given current time. The response should follows the following requirements. \n
    ### Requirements:
    1. Your response should consist only of an accurate time in the format 'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.' 
    2. Please carefully check the accuracy of the identifiction results. 
    3. The phrase "in the last month" means "in the thirty or so days up to and including today".\n
    ### Current Time:\n{}\n
    ### User Query:\n{}\n
    ### Response:\n""".format(cur_time, query)

    return prompt


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


def set_cpu_running_env():
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"