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

import uuid
from typing import Optional, Union, Dict, Any

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, InvalidPasswordException
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from httpx_oauth.clients.github import GitHubOAuth2
from httpx_oauth.clients.facebook import FacebookOAuth2
from httpx_oauth.clients.twitter import TwitterOAuth1

from ..database.user_db import User, get_user_db
from ..schemas.user import UserCreate

from ..config.config import get_settings

from intel_extension_for_transformers.utils import logger

global_settings = get_settings()

SECRET = "SECRET"

google_oauth_client = GoogleOAuth2(
    global_settings.google_oauth_client_id,
    global_settings.google_oauth_client_secret
)

github_oauth_client = GitHubOAuth2(
    global_settings.github_oauth_client_id,
    global_settings.github_oauth_client_secret
)

facebook_oauth_client = FacebookOAuth2(
    global_settings.facebook_oauth_client_id,
    global_settings.facebook_oauth_client_secret
)

twitter_oauth_client = TwitterOAuth1(
    global_settings.twitter_oauth_client_id,
    global_settings.twitter_oauth_client_secret
)

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def validate_password(
        self,
        password: str,
        user: Union[UserCreate, User],
    ) -> None:
        if len(password) < 8:
            raise InvalidPasswordException(
                reason="Password should be at least 8 characters"
            )
        if user.email in password:
            raise InvalidPasswordException(
                reason="Password should not contain e-mail"
            )

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"Verification requested for user {user.id}. Verification token: {token}")

    async def on_after_update(
        self,
        user: User,
        update_dict: Dict[str, Any],
        request: Optional[Request] = None,
    ):
        logger.info(f"User {user.id} has been updated with {update_dict}.")

    async def on_after_login(
        self,
        user: User,
        request: Optional[Request] = None,
    ):
        logger.info(f"User {user.id} logged in.")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"Verification requested for user {user.id}. Verification token: {token}")

    async def on_after_verify(
        self, user: User, request: Optional[Request] = None
    ):
        logger.info(f"User {user.id} has been verified")

    async def on_after_reset_password(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} has reset their password.")

    async def on_before_delete(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} is going to be deleted")

    async def on_after_delete(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} is successfully deleted")

    async def get_oauth2_client(self, provider: str):
        if provider == "google":
            return google_oauth_client
        elif provider == "github":
            return github_oauth_client
        elif provider == "facebook":
            return facebook_oauth_client
        elif provider == "twitter":
            return twitter_oauth_client
        else:
            raise ValueError("Unsupported OAuth provider")

    async def get_oauth2_login_url(self, provider: str, state: str):
        client = await self.get_oauth2_client(provider)
        return client.get_authorization_url(state=state)

    async def on_after_login_with_oauth2(
        self, user: User, provider: str, request: Optional[Request] = None
    ):
        logger.info(f"User {user.id} logged in with {provider.capitalize()}.")

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


auth_backends = [
    AuthenticationBackend(
        name="jwt",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    ),
    AuthenticationBackend(
        name="google",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    ),
    AuthenticationBackend(
        name="github",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    ),
    AuthenticationBackend(
        name="facebook",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    ),
    AuthenticationBackend(
        name="twitter",
        transport=bearer_transport,
        get_strategy=get_jwt_strategy,
    ),
]

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager, auth_backends
)

current_active_user = fastapi_users.current_user(active=True)

