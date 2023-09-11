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

"""
Merge two conversation files into one

Usage: python3 -m fastchat.data.merge --in file1.json file2.json --out merged.json
"""

import argparse
import json
from typing import Dict, Sequence, Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True, nargs="+")
    parser.add_argument("--out-file", type=str, default="merged.json")
    args = parser.parse_args()

    new_content = []
    for in_file in args.in_file:
        content = json.load(open(in_file, "r"))
        new_content.extend(content)

    json.dump(new_content, open(args.out_file, "w"), indent=2)
