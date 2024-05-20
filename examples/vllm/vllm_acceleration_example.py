#  Copyright (c) 2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
from typing import List, Optional
from intel_extension_for_transformers.transformers import AutoModelForCausalLM


def main(args_in: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to a NE compatible file")
    parser.add_argument("--model_path", type=str, help="Model name: String", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt to start generation with: String (default: empty)",
        default="Once upon a time",
    )
    args = parser.parse_args(args_in)
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_vllm=True)
    output = model.generate(args.prompt)
    print(output)


if __name__ == "__main__":
    main()
