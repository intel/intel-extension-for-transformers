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

from .config import get_settings
from contextlib import contextmanager

global_settings = get_settings()

class MysqlDb(object):
    def __init__(self):
        self._host = global_settings.mysql_host
        self._port = global_settings.mysql_port
        self._db = global_settings.mysql_db
        self._user = global_settings.mysql_user
        self._passwd = global_settings.mysql_password
        self._charset = 'utf8'
        self._connect()

    def _connect(self):
        from pymysql import connect, cursors
        self._conn = connect(host=self._host,
                             port=self._port,
                             user=self._user,
                             passwd=self._passwd,
                             db=self._db,
                             charset=self._charset,
                             cursorclass=cursors.DictCursor)
        self._cursor = self._conn.cursor()

    def _set_db(self, db):
        self._close()
        self._db = db
        self._connect()

    def _close(self):
        self._cursor.close()
        self._conn.close()

    @contextmanager
    def transaction(self):
        try:
            yield
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e

    def fetch_one(self, sql, params=None):
        from pymysql.converters import escape_string
        escape_sql = escape_string(sql)
        print(f"escape sql: {escape_sql}")
        self._cursor.execute(escape_sql, params)
        return self._cursor.fetchone()

    def fetch_all(self, sql, params=None):
        from pymysql.converters import escape_string
        escape_sql = escape_string(sql)
        print(f"escape sql: {escape_sql}")
        self._cursor.execute(escape_sql, params)
        return self._cursor.fetchall()

    def insert(self, sql, params):
        return self._edit(sql, params)

    def update(self, sql, params):
        return self._edit(sql, params)

    def delete(self, sql, params):
        return self._edit(sql, params)

    def _edit(self, sql, params):
        return self._cursor.execute(sql, params)
