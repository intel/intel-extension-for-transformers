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

import argparse
import json

import requests

from fastchat.conversation import (
    get_default_conv_template,
    compute_skip_echo_len,
    SeparatorStyle,
)


def main():
    model_name = args.model_name

    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/refresh_all_workers")
        ret = requests.post(controller_addr + "/list_models")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")

        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        return

    conv = get_default_conv_template(model_name).copy()
    conv.append_message(conv.roles[0], args.message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    headers = {"User-Agent": "fastchat Client"}
    pload = {
        "model": model_name,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=pload,
        stream=True,
    )

    print(f"{conv.roles[0]}: {args.message}")
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            skip_echo_len = compute_skip_echo_len(model_name, conv, prompt)
            output = data["text"][skip_echo_len:].strip()
            print(f"{conv.roles[1]}: {output}", end="\r")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--message", type=str, default="Tell me a story with more than 1000 words."
    )
    args = parser.parse_args()

    main()
