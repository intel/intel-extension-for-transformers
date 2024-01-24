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
import contextlib
import asyncio
import os
from fastapi.testclient import TestClient
from intel_extension_for_transformers.neural_chat.server.neuralchat_server import app, setup_authentication_router
from intel_extension_for_transformers.neural_chat.server.database.user_db import create_db_and_tables, get_async_session, get_user_db
from intel_extension_for_transformers.neural_chat.server.schemas.user import UserCreate
from intel_extension_for_transformers.neural_chat.server.user.users import get_user_manager
from fastapi_users.exceptions import UserAlreadyExists

get_async_session_context = contextlib.asynccontextmanager(get_async_session)
get_user_db_context = contextlib.asynccontextmanager(get_user_db)
get_user_manager_context = contextlib.asynccontextmanager(get_user_manager)


async def create_user(email: str, password: str, is_superuser: bool = False):
    try:
        async with get_async_session_context() as session:
            async with get_user_db_context(session) as user_db:
                async with get_user_manager_context(user_db) as user_manager:
                    user = await user_manager.create(
                        UserCreate(
                            email=email, password=password, is_superuser=is_superuser
                        )
                    )
                    print(f"User created {user}")
    except UserAlreadyExists:
        print(f"User {email} already exists")


class TestAuthenticationRouter(unittest.TestCase):
    def setUp(self):
        setup_authentication_router()
        asyncio.run(create_db_and_tables())
        self.client = TestClient(app)
        asyncio.run(create_user("test_user@example.com", "test_password"))

    def tearDown(self) -> None:
        if os.path.exists('users.db'):
            os.remove('users.db')
        return super().tearDown()

    def test_register_router(self):
        response = self.client.post("/auth/register", json={"email": "new_user@example.com", "password": "password"})
        self.assertEqual(response.status_code, 201)

    def test_auth_jwt_router(self):
        response = self.client.post("/auth/jwt/login", data={"username": "test_user@example.com", "password": "test_password"})
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
