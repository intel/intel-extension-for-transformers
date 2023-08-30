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
Do optional cleaning (e.g., remove some languages).

Usage:
python3 -m fastchat.data.optional_clean --in input.json --out output.json --keep-lang en
python3 -m fastchat.data.optional_clean --in input.json --out output.json --skip-lang en

Requirement:
pip3 install polyglot icu pyicu pycld2 morfessor
"""
import argparse
import json
import re

import polyglot
from polyglot.detect import Detector
import pycld2
from tqdm import tqdm


def skip(conv, args):
    # Remove certain languages
    if args.keep_lang != "all" or args.skip_lang is not None:
        text = "\n".join([x["value"] for x in conv["conversations"]])
        try:
            lang_code = Detector(text).language.code
        except (pycld2.error, polyglot.detect.base.UnknownLanguage):
            lang_code = "unknown"

        if args.keep_lang != "all" and lang_code != args.keep_lang:
            return True

        if lang_code == args.skip_lang:
            return True

    # Remove repetitive numbers
    if args.reduce_rep:
        for sentence in conv["conversations"]:
            val = sentence["value"]
            sub = re.search(r"(\d)\1{8}", val)
            if sub is not None:
                return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str)
    parser.add_argument(
        "--keep-lang",
        type=str,
        default="all",
        choices=["all", "en"],
        help="Only keep certain langauges.",
    )
    parser.add_argument("--skip-lang", type=str, help="Skip a specific language.")
    # NOTE: Be careful about reduce_rep which may remove some good data.
    # For example, addresses could have long consecutive 0's
    parser.add_argument("--reduce-rep", action="store_true")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    keep_lang = args.keep_lang
    skip_lang = args.skip_lang
    reduce_rep = args.reduce_rep
    assert keep_lang == "all" or skip_lang is None

    if out_file is None:
        out_file = "sharegpt_clean"
        if keep_lang != "all":
            out_file += "_" + keep_lang
        if skip_lang is not None:
            out_file += "_skip_" + skip_lang
        if reduce_rep:
            out_file += "_reduce_rep"
        out_file += ".json"

    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    for conv in tqdm(content):
        if not skip(conv, args):
            new_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)
