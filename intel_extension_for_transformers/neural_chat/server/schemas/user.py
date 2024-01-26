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

from fastapi_users import schemas
from typing import Optional


class UserRead(schemas.BaseUser[uuid.UUID]):
    role: str
    is_vipuser: bool = False
    wwid: Optional[str] = None
    email_address: Optional[str] = None
    account: Optional[str] = None
    name: Optional[str] = None
    given_name: Optional[str] = None
    distinguished_name: Optional[str] = None
    idsid: Optional[str] = None
    generic: bool = False
    SuperGroup: Optional[str] = None
    Group: Optional[str] = None
    Division: Optional[str] = None
    DivisionLong: Optional[str] = None
    CostCenterLong: Optional[str] = None
    mgrWWID: Optional[str] = None

class UserCreate(schemas.BaseUserCreate):
    role: Optional[str] = 'user'
    is_vipuser: bool = False
    wwid: Optional[str] = None
    email_address: Optional[str] = None
    account: Optional[str] = None
    name: Optional[str] = None
    given_name: Optional[str] = None
    distinguished_name: Optional[str] = None
    idsid: Optional[str] = None
    generic: bool = False
    SuperGroup: Optional[str] = None
    Group: Optional[str] = None
    Division: Optional[str] = None
    DivisionLong: Optional[str] = None
    CostCenterLong: Optional[str] = None
    mgrWWID: Optional[str] = None

class UserUpdate(schemas.BaseUserUpdate):
    role: Optional[str]
    is_vipuser: Optional[bool]
    wwid: Optional[str] = None
    email_address: Optional[str] = None
    account: Optional[str] = None
    name: Optional[str] = None
    given_name: Optional[str] = None
    distinguished_name: Optional[str] = None
    idsid: Optional[str] = None
    generic: bool = False
    SuperGroup: Optional[str] = None
    Group: Optional[str] = None
    Division: Optional[str] = None
    DivisionLong: Optional[str] = None
    CostCenterLong: Optional[str] = None
    mgrWWID: Optional[str] = None