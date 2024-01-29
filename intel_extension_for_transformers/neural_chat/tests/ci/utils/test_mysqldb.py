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


import unittest
from unittest.mock import patch, MagicMock
from intel_extension_for_transformers.neural_chat.utils.database.mysqldb import MysqlDb


class TestMysqlDb(unittest.TestCase):

    @patch('pymysql.connect')
    def test_mysql_db_methods(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = {'image_id': '1'}
        mock_cursor.fetchall.return_value = [{'image_id': '1'}, {'image_id': '2'}]
        mock_cursor.execute.return_value = True

        db = MysqlDb()
        one_result = db.fetch_one("SELECT * FROM table WHERE id = 1")
        all_result = db.fetch_all("SELECT * FROM table")
        insert_result = db.insert("INSERT INTO table mock_table VALUES ('mock_value')", None)

        self.assertEqual(one_result,  {'image_id': '1'})
        self.assertEqual(len(all_result), 2)
        self.assertTrue(insert_result)

        mock_cursor.fetchone.assert_called_once()
        mock_cursor.fetchall.assert_called_once()
        mock_cursor.execute.assert_called()


if __name__ == "__main__":
    unittest.main()
