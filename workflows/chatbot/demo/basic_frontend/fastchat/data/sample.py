# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample some conversations from a file.

Usage: python3 -m fastchat.data.sample --in sharegpt.json --out sampled.json
"""
import argparse
import json
from typing import Dict, Sequence, Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sampled.json")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    content = json.load(open(args.in_file, "r"))
    new_content = []
    for i in range(args.begin, args.end):
        sample = content[i]
        concat = ""
        for s in sample["conversations"]:
            concat += s["value"]

        if len(concat) > args.max_length:
            continue

        new_content.append(sample)

    json.dump(new_content, open(args.out_file, "w"), indent=2)
