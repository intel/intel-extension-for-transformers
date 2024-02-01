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


import json
import datetime
from datetime import timedelta, timezone
from typing import Dict
from .database.mysqldb import MysqlDb


def record_request(request_url: str, request_body: Dict, user_id: str):
    mysqldb = MysqlDb()
    mysqldb._set_db("requests")
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_time = utc_now.astimezone(SHA_TZ).strftime("%Y-%m-%d %H:%M:%S")
    if not isinstance(request_body, Dict):
        request_body = request_body.__dict__
    request_body = json.dumps(request_body).replace("'", "^")
    sql = f"""INSERT INTO record VALUES \
        (null, '{request_url}', '{request_body}', '{user_id}', '{beijing_time}');"""
    print(f"[record request] sql: {sql}")
    try:
        with mysqldb.transaction():
            mysqldb.insert(sql, None)
    except Exception as e:
        raise Exception(f"[record request] Exception occurred: {e}")
    mysqldb._close()
